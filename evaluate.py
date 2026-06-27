import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from utils.dataset import GoProDataset
from models.model import SimpleCNN, UNet, ResNetUNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_model(model, loader, device):
    model.eval()
    psnr_scores, ssim_scores = [], []
    
    with torch.no_grad():
        for blur, sharp in loader:
            blur, sharp = blur.to(device), sharp.to(device)
            output = model(blur).clamp(0, 1)
            
            # Convert to numpy (H, W, C) for skimage
            out_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            sharp_np = sharp.squeeze().cpu().numpy().transpose(1, 2, 0)
            
            psnr_scores.append(psnr(sharp_np, out_np))
            ssim_scores.append(ssim(sharp_np, out_np, channel_axis=2, data_range=1.0))
            
    return np.mean(psnr_scores), np.mean(ssim_scores)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = Compose([Resize((256, 256)), ToImage(), ToDtype(torch.float32, scale=True)])
    
    # Load test data
    test_ds = GoProDataset('data/test', transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1)
    
    # Define your model zoo
    model_zoo = [
        ("CNN (MSE)", SimpleCNN(), "checkpoints/cnn_final.pth"),
        ("U-Net (VGG)", UNet(), "checkpoints/unet_vgg.pth"),
        ("U-Net (Final)", UNet(), "checkpoints/unet_final.pth"),
        ("ResNet (VGG)", ResNetUNet(), "checkpoints/resnet_vgg.pth"),
        ("ResNet (GAN)", ResNetUNet(), "checkpoints/resnet_gan.pth"),
    ]
    
    print(f"{'Model':<20} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 45)
    
    for name, model, path in model_zoo:
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            p, s = evaluate_model(model, test_loader, device)
            print(f"{name:<20} | {p:<10.2f} | {s:<10.4f}")
        except FileNotFoundError:
            print(f"{name:<20} | File not found")

if __name__ == '__main__':
    main()