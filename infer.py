import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import os
import argparse
from models.model import SimpleCNN, UNet

# 1. Setup arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='unet', help='cnn or unet')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Initialize correct model
if args.model == 'cnn':
    model = SimpleCNN().to(device)
    checkpoint_path = "checkpoints/cnn_final.pth"
else:
    model = UNet().to(device)
    checkpoint_path = "checkpoints/unet_final.pth"

# 3. Load weights
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded weights from {checkpoint_path}")
else:
    print(f"Error: {checkpoint_path} not found. Did you train it yet?")
    exit()

model.eval()

# 4. Inference setup
blur_dir = "data/train/blur"
first_image_name = sorted(os.listdir(blur_dir))[0]
test_img_path = os.path.join(blur_dir, first_image_name)

print(f"Testing with image: {first_image_name} using {args.model}")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

# 5. Load and process
blur_img = Image.open(test_img_path).convert("RGB")
input_tensor = transform(blur_img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

# 6. Save
output_img = output.squeeze(0).cpu().clamp(0, 1)
output_img = transforms.ToPILImage()(output_img)
output_img.save(f"deblurred_result_{args.model}.png")

print(f"Result saved as 'deblurred_result_{args.model}.png'.")