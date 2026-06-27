import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import argparse
from utils.dataset import get_loaders
# 1. Added ResNetUNet to imports
from models.model import SimpleCNN, UNet, ResNetUNet 
from models.loss import VGGPerceptualLoss

def get_model(model_name):
    # 2. Logic to select the model
    if model_name == 'cnn': return SimpleCNN()
    if model_name == 'unet': return UNet()
    if model_name == 'resnet': return ResNetUNet()
    raise ValueError(f"Unknown model name: {model_name}")

def train(model_name, use_vgg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True)
    ])
    train_loader, val_loader, _ = get_loaders('data', transform)
    
    # 3. Use the get_model helper
    model = get_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    mse_criterion = nn.MSELoss()
    vgg_criterion = VGGPerceptualLoss().to(device) if use_vgg else None

    print(f"Starting training: {model_name} | VGG={use_vgg}")
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        for blur, sharp in train_loader:
            blur, sharp = blur.to(device), sharp.to(device)
            optimizer.zero_grad()
            output = model(blur)
            
            # Loss Calculation
            loss = mse_criterion(output, sharp)
            if use_vgg:
                loss += 0.01 * vgg_criterion(output, sharp)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{50} | {model_name} | Loss: {avg_loss:.6f}")

    save_name = f"checkpoints/{model_name}_{'vgg' if use_vgg else 'mse'}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Model saved to {save_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 4. Now you can use --model resnet
    parser.add_argument('--model', type=str, default='unet', help='cnn, unet, or resnet')
    parser.add_argument('--loss', type=str, default='mse', help='mse or vgg')
    args = parser.parse_args()
    train(args.model, args.loss == 'vgg')