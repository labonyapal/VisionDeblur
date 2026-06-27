import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # Use VGG16_Weights.DEFAULT instead of pretrained=True
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.slice.parameters():
            param.requires_grad = False
            
    def forward(self, input, target):
        input_feat = self.slice(input)
        target_feat = self.slice(target)
        return nn.functional.mse_loss(input_feat, target_feat)