import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ImageNetDataLoader:
    def __init__(self, data_dir, batch_size=256, num_workers=4):
        """
        Initialize ImageNet data loader
        Args:
            data_dir: Path to ImageNet-100 directory containing 'train' and 'val' folders
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # ImageNet normalization values
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Verify dataset directory structure
        self._verify_dataset()
        
    def _verify_dataset(self):
        """Verify that the dataset directory has the correct structure"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
            
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
            
        # Verify that we have the correct number of classes
        train_classes = len(os.listdir(train_dir))
        val_classes = len(os.listdir(val_dir))
        
        if train_classes != 100 or val_classes != 100:
            print(f"Warning: Expected 100 classes, but found {train_classes} in train and {val_classes} in val")
        
    def get_transforms(self, train=True):
        """
        Get data transformations
        Args:
            train: If True, return training transforms with augmentation
        """
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
    
    def get_loaders(self):
        """
        Get train and validation data loaders
        Returns:
            train_loader, val_loader: DataLoader objects for training and validation
        """
        # Training dataset
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=self.get_transforms(train=True)
        )
        
        # Validation dataset
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"),
            transform=self.get_transforms(train=False)
        )
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Test the data loader
    data_dir = "./imagenet/imagenet100"  # Path to your local ImageNet-100 directory
    loader = ImageNetDataLoader(data_dir)
    train_loader, val_loader = loader.get_loaders()
    
    # Print some information about the dataset
    print("\nTesting data loader:")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break 