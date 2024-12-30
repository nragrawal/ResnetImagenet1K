import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import time
import copy
from network import create_model
from calculate_stats import calculate_mean_std
from verify_dataset import verify_dataset_structure
from datetime import datetime
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel
# Replace torchsummary import with torchinfo
from torchinfo import summary

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class ImageNetSubsetLoader:
    def __init__(self, data_dir, batch_size=256, num_workers=4):
        """
        Initialize ImageNet subset data loader
        Args:
            data_dir: Path to ILSVRC subset directory
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        # Data directories
        train_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'train')
        val_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'val')
        
        # Load dataset statistics
        stats_file = os.path.join(data_dir, 'dataset_stats.txt')
        with open(stats_file, 'r') as f:
            self.mean = eval(f.readline())
            self.std = eval(f.readline())
        
        # Create datasets
        train_dataset = datasets.ImageFolder(
            train_dir,
            transform=self.get_transforms(train=True)
        )
        
        val_dataset = datasets.ImageFolder(
            val_dir,
            transform=self.get_transforms(train=False)
        )
        
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.num_classes = len(train_dataset.classes)
    
    def get_transforms(self, train=True):
        """Get data transforms using dataset statistics"""
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

class Trainer:
    def __init__(self, config):
        # Create output directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Set up logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join('outputs', f'training_log_{timestamp}.txt')
        sys.stdout = Logger(self.log_file)
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create data loader
        data_loader = ImageNetSubsetLoader(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        self.train_loader = data_loader.train_loader
        self.val_loader = data_loader.val_loader
        self.num_classes = data_loader.num_classes
        
        # Initialize starting epoch and best accuracy
        self.start_epoch = 1
        self.best_acc = 0.0
        self.best_model_wts = None
        
        # Create model and move to device
        self.model = create_model(
            num_classes=self.num_classes,
            pretrained=config['pretrained']
        )
        
        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            # Wrap model with DataParallel
            self.model = DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Initialize mixed precision training
        self.scaler = amp.GradScaler()
        self.autocast = amp.autocast
        print("Using mixed precision training")
        
        # Resume from checkpoint if specified
        if config['resume']:
            if os.path.isfile(config['resume']):
                print(f"Loading checkpoint '{config['resume']}'")
                checkpoint = torch.load(config['resume'])
                
                # Handle DataParallel state dict
                if torch.cuda.device_count() > 1:
                    # If checkpoint was saved with DataParallel
                    if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # If checkpoint was saved without DataParallel
                        new_state_dict = {f'module.{k}': v for k, v in checkpoint['model_state_dict'].items()}
                        self.model.load_state_dict(new_state_dict)
                else:
                    # If using single GPU but checkpoint was saved with DataParallel
                    if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
                        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
                        self.model.load_state_dict(new_state_dict)
                    else:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer and scheduler states
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Set starting epoch and best accuracy
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_acc = checkpoint['best_acc']
                
                print(f"Loaded checkpoint '{config['resume']}' (epoch {checkpoint['epoch']})")
                print(f"Previous best accuracy: {self.best_acc*100:.2f}%")
            else:
                print(f"No checkpoint found at '{config['resume']}'")
        
        # Print model summary after initialization
        print("\nModel Summary:")
        summary(self.model, 
                input_size=(config['batch_size'], 3, 224, 224),
                device=self.device,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        print("\n")
    
    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Use autocast for mixed precision training
            with self.autocast():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
            
            # Scale loss and do backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_data += inputs.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss/processed_data:.4f}',
                'acc': f'{(running_corrects.double()/processed_data)*100:.2f}%'
            })
        
        epoch_loss = running_loss / processed_data
        epoch_acc = running_corrects.double() / processed_data
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_top1_corrects = 0
        running_top5_corrects = 0
        processed_data = 0
        
        pbar = tqdm(self.val_loader, desc='Validating')
        with torch.no_grad(), self.autocast():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Calculate top-1 and top-5 accuracy
                _, pred = outputs.topk(5, 1, True, True)
                labels = labels.view(-1, 1).expand_as(pred)  # Shape: batch_size x 5
                correct = pred.eq(labels)
                
                # Top-1 accuracy
                top1_correct = correct[:, 0]
                running_top1_corrects += top1_correct.sum().item()
                
                # Top-5 accuracy
                top5_correct = correct.any(dim=1)
                running_top5_corrects += top5_correct.sum().item()
                
                running_loss += loss.item() * inputs.size(0)
                processed_data += inputs.size(0)
                
                pbar.set_postfix({
                    'loss': f'{running_loss/processed_data:.4f}',
                    'top1': f'{(running_top1_corrects/processed_data)*100:.2f}%',
                    'top5': f'{(running_top5_corrects/processed_data)*100:.2f}%'
                })
        
        epoch_loss = running_loss / processed_data
        top1_acc = running_top1_corrects / processed_data
        top5_acc = running_top5_corrects / processed_data
        
        return epoch_loss, top1_acc, top5_acc
    
    def train(self):
        print(f"Training on {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Logging to: {self.log_file}")
        
        since = time.time()
        
        # Start from the loaded epoch if resuming
        for epoch in range(self.start_epoch, self.config['epochs'] + 1):
            # Training phase
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validation phase
            val_loss, val_top1_acc, val_top5_acc = self.validate()
            
            # Step the scheduler
            self.scheduler.step()
            
            # Print and log epoch summary
            summary = f"\nEpoch {epoch}/{self.config['epochs']}:\n"
            summary += f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}%\n"
            summary += f"Val Loss: {val_loss:.4f} Top-1: {val_top1_acc*100:.2f}% Top-5: {val_top5_acc*100:.2f}%"
            print(summary)
            
            # Save best model based on top-1 accuracy
            is_best = val_top1_acc > self.best_acc
            if is_best:
                self.best_acc = val_top1_acc
                print(f'New best accuracy: Top-1 {self.best_acc*100:.2f}%')
            
            # Save checkpoint every epoch
            self.save_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_top1_acc=val_top1_acc,
                val_top5_acc=val_top5_acc,
                is_best=is_best
            )
        
        time_elapsed = time.time() - since
        final_summary = f'\nTraining completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s\n'
        final_summary += f'Best val accuracy: {self.best_acc*100:.2f}%'
        print(final_summary)
        
        # Load best model weights
        self.model.load_state_dict(torch.load(os.path.join(self.config['checkpoint_dir'], 'model_best.pth'))['model_state_dict'])
        return self.model

    def save_checkpoint(self, epoch, train_loss, train_acc, val_loss, val_top1_acc, val_top5_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_top1_acc': val_top1_acc,
            'val_top5_acc': val_top5_acc,
            'best_acc': self.best_acc,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch:03d}.pth'  # Zero-padded epoch number
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'model_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to: {best_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet on ImageNet subset')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to ILSVRC subset directory')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=90,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1,
                      help='Initial learning rate')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained model')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Verify dataset before training
    print("Verifying dataset structure...")
    if not verify_dataset_structure(args.data_dir):
        print("Dataset verification failed. Please fix the issues before training.")
        exit(1)
    
    # Adjust batch size for multiple GPUs
    num_gpus = torch.cuda.device_count()
    effective_batch_size = args.batch_size
    
    if num_gpus > 1:
        # Scale batch size by number of GPUs
        args.batch_size = args.batch_size // num_gpus
        print(f"Scaling batch size to {args.batch_size} per GPU "
              f"(effective batch size: {effective_batch_size})")
    
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,  # Per-GPU batch size
        'num_workers': 4 * num_gpus,    # Scale workers with GPUs
        'learning_rate': args.lr * num_gpus,  # Scale learning rate with GPUs
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'pretrained': args.pretrained,
        'checkpoint_dir': 'checkpoints',
        'resume': args.resume
    }
    
    trainer = Trainer(config)
    model = trainer.train() 