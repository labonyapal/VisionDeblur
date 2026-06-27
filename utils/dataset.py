import torch
import os
from torch.utils.data import Dataset, random_split
from PIL import Image

class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.blur_path = os.path.join(root_dir, 'blur')
        self.sharp_path = os.path.join(root_dir, 'sharp')
        
        # FIX: Filter list to include ONLY files, not subdirectories
        all_items = os.listdir(self.blur_path)
        self.files = sorted([f for f in all_items if os.path.isfile(os.path.join(self.blur_path, f))])
        
        self.transform = transform

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        blur = Image.open(os.path.join(self.blur_path, self.files[idx])).convert('RGB')
        sharp = Image.open(os.path.join(self.sharp_path, self.files[idx])).convert('RGB')
        
        if self.transform: 
            blur = self.transform(blur)
            sharp = self.transform(sharp)
        return blur, sharp

def get_loaders(root_dir, transform, batch_size=4):
    full_train = GoProDataset(os.path.join(root_dir, 'train'), transform=transform)
    
    # Split training data into 90% train, 10% validation
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    test_ds = GoProDataset(os.path.join(root_dir, 'test'), transform=transform)
    
    # Return loaders
    return [
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    ]