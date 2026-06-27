import io
import os
import base64
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Import the model classes exactly as they are defined in the repository
from models.model import SimpleCNN, UNet, ResNetUNet

app = FastAPI(title="Image Deblurring Comparison API")

# Enable CORS for React frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production, but for local development, * is standard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = {}

# Model definitions and paths matching the checkpoints directory
MODEL_SPECS = {
    "cnn": {
        "class": SimpleCNN,
        "path": "checkpoints/cnn_final.pth",
        "display_name": "CNN"
    },
    "unet_final": {
        "class": UNet,
        "path": "checkpoints/unet_final.pth",
        "display_name": "UNet"
    },
    "unet_vgg": {
        "class": UNet,
        "path": "checkpoints/unet_vgg.pth",
        "display_name": "UNet + VGG"
    },
    "resnet_vgg": {
        "class": ResNetUNet,
        "path": "checkpoints/resnet_vgg.pth",
        "display_name": "ResNet + VGG"
    },
    "resnet_gan": {
        "class": ResNetUNet,
        "path": "checkpoints/resnet_gan_best.pth",
        "display_name": "ResNet + GAN"
    }
}

@app.on_event("startup")
def load_checkpoints():
    print(f"Initializing models on device: {device}")
    for name, spec in MODEL_SPECS.items():
        try:
            model = spec["class"]().to(device)
            # Load the checkpoint
            state_dict = torch.load(spec["path"], map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            loaded_models[name] = model
            print(f"Loaded {spec['display_name']} from {spec['path']}")
        except Exception as e:
            print(f"WARNING: Failed to load {name} model from {spec['path']}: {e}")

def compute_laplacian_variance(pil_img: Image.Image) -> float:
    """
    Computes Laplacian variance on a 0-255 grayscale scale using PyTorch Conv2D.
    This matches the standard OpenCV cv2.Laplacian().var() calculation.
    """
    gray_img = pil_img.convert("L")
    img_np = np.array(gray_img, dtype=np.float32)
    
    with torch.no_grad():
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        # Standard Laplacian kernel
        kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, -4.0, 1.0],
                               [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        # Apply convolution
        laplacian = F.conv2d(img_tensor, kernel, padding=1)
        variance = laplacian.var().item()
        return variance

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Converts a [3, H, W] float32 tensor in range [0, 1] to a base64 PNG data URL."""
    # Ensure it's on CPU and clamped
    clamped = torch.clamp(tensor, 0.0, 1.0).cpu()
    img_pil = transforms.ToPILImage()(clamped)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.post("/deblur")
async def deblur_image(
    file: UploadFile = File(None),
    use_sample: bool = Query(False)
):
    pil_img = None
    filename = "sample_image.png"

    if file is None or use_sample:
        # Load sample image fallback
        test_blur_dir = "data/test/blur"
        if not os.path.exists(test_blur_dir):
            raise HTTPException(status_code=400, detail="Sample image directory not found (data/test/blur)")
        images = sorted(os.listdir(test_blur_dir))
        if not images:
            raise HTTPException(status_code=400, detail="No sample images found in data/test/blur")
        
        sample_img_path = os.path.join(test_blur_dir, images[0])
        try:
            pil_img = Image.open(sample_img_path).convert("RGB")
            filename = images[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load sample image: {str(e)}")
    else:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        try:
            # Load image with PIL
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            filename = file.filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # 1. Compute blur severity using Laplacian variance (on original scale)
    lap_var = compute_laplacian_variance(pil_img)
    
    # Severity classification: above 500 = sharp, 100-500 = mild, 50-100 = medium, below 50 = heavy
    if lap_var > 500:
        blur_level = "SHARP"
    elif lap_var >= 100:
        blur_level = "MILD"
    elif lap_var >= 50:
        blur_level = "MEDIUM"
    else:
        blur_level = "HEAVY"

    # 2. Resize and normalize image for model input
    # Standard resize to 256x256, normalize to [0,1] range
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)  # (1, 3, 256, 256)
    
    # Save the base64 of the original blurry image (resized to 256x256 for perfect alignment)
    original_base64 = tensor_to_base64(input_tensor.squeeze(0))

    results = {}
    
    # 3. Run inference on all 5 models
    for name, spec in MODEL_SPECS.items():
        if name not in loaded_models:
            # Skip if model failed to load at startup
            continue
            
        model = loaded_models[name]
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
            
        # Compute PSNR and SSIM comparing output to the original blurry input (resized to 256x256)
        # PSNR Calculation:
        mse = torch.mean((output_tensor - input_tensor) ** 2).item()
        psnr_score = 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))
        
        # SSIM Calculation:
        out_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        in_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ssim_score = float(ssim(in_np, out_np, channel_axis=2, data_range=1.0))
        
        # Convert output to base64
        output_base64 = tensor_to_base64(output_tensor.squeeze(0))
        
        results[name] = {
            "image": output_base64,
            "psnr": round(psnr_score, 2),
            "ssim": round(ssim_score, 4)
        }

    return {
        "blur_level": blur_level,
        "laplacian_var": round(lap_var, 2),
        "original_image": original_base64,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
