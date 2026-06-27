import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from models.model import ResNetUNet, PatchGANDiscriminator
from models.loss import VGGPerceptualLoss
from utils.dataset import get_loaders

def train_gan():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    
    gen  = ResNetUNet().to(device)
    disc = PatchGANDiscriminator().to(device)
    
    # Fix 1: D learns 10x slower than G
    opt_g = optim.Adam(gen.parameters(),  lr=1e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=1e-5, betas=(0.5, 0.999))
    
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()
    vgg           = VGGPerceptualLoss().to(device)
    
    # Load pretrained ResNet weights as starting point
    gen.load_state_dict(torch.load("checkpoints/resnet_vgg.pth", map_location=device))
    print("Loaded pretrained ResNet weights")
    
    train_loader, _, _ = get_loaders('data', transform)
    print("Starting Stable GAN training...")
    
    best_g_loss = float('inf')
    
    for epoch in range(50):
        
        # Fix: initialize so print never crashes
        loss_d = torch.tensor(0.0)
        loss_g = torch.tensor(0.0)
        
        total_d, total_g = 0.0, 0.0
        
        for i, (blur, sharp) in enumerate(train_loader):
            blur, sharp = blur.to(device), sharp.to(device)
            
            # --- Train Discriminator every 2 batches ---
            if i % 2 == 0:
                opt_d.zero_grad()
                
                with torch.no_grad():
                    fake = gen(blur)
                
                pred_real = disc(sharp)
                pred_fake = disc(fake.detach())
                
                # Fix 2: Label smoothing
                loss_d = (
                    criterion_gan(pred_real, torch.full_like(pred_real, 0.9)) +
                    criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                ) * 0.5
                
                loss_d.backward()
                opt_d.step()
                total_d += loss_d.item()
            
            # --- Train Generator every batch ---
            opt_g.zero_grad()
            fake = gen(blur)
            pred_fake = disc(fake)
            
            # Fix 3: Balanced loss — GAN + VGG + L1
            loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_vgg = vgg(fake, sharp)
            loss_l1  = criterion_l1(fake, sharp)
            
            loss_g = (1.0  * loss_gan +
                      1.0  * loss_vgg +
                      10.0 * loss_l1)
            
            loss_g.backward()
            opt_g.step()
            total_g += loss_g.item()
        
        avg_d = total_d / (len(train_loader) // 2)
        avg_g = total_g / len(train_loader)
        
        print(f"Epoch {epoch+1:02d} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")
        
        # Save best generator
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            torch.save(gen.state_dict(), "checkpoints/resnet_gan_best.pth")
            print(f"  → Best model saved at epoch {epoch+1}")
    
    # Save final model too
    torch.save(gen.state_dict(), "checkpoints/resnet_gan.pth")
    print("GAN training complete.")

if __name__ == '__main__':
    train_gan()