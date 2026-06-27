import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 3, 3, padding=1)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1))
    def forward(self, x): return self.dec(self.enc(x))

# --- Phase 4: Residual Blocks ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        # The core of a ResNet: add the original input 'x' 
        # to the output of the convolutional layers
        return x + self.block(x)

class ResNetUNet(nn.Module):
    def __init__(self):
        super(ResNetUNet, self).__init__()
        # Initial projection
        self.enc = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
        
        # Deep stack of 5 Residual Blocks for feature learning
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        
        # Reconstruction layer
        self.dec = nn.Sequential(nn.Conv2d(64, 3, 3, padding=1))
        
    def forward(self, x):
        identity = self.enc(x)
        out = self.res_blocks(identity)
        # Adding the skip connection from the input of the res-blocks
        return self.dec(out + identity)
    
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, padding=1) # Outputs a patch of probabilities
        )
    def forward(self, x): return self.net(x)