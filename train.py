import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import copy
from data_loader import ImageNetDataLoader
from network import create_model as create_model_imagenet1000
from network_imagenet100 import create_model_imagenet100

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data loaders
        data_loader = ImageNetDataLoader(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        self.train_loader, self.val_loader = data_loader.get_loaders()
        
        # Create model based on dataset type
        if config['dataset'] == 'imagenet100':
            self.model = create_model_imagenet100(pretrained=config['pretrained'])
        else:  # imagenet1000
            self.model = create_model_imagenet1000(pretrained=config['pretrained'])
            
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        self.best_acc = 0.0
        self.best_model_wts = None

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}')
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_data += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / processed_data,
                'acc': running_corrects.double() / processed_data
            })
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
        
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0
        
        pbar = tqdm(self.val_loader, desc='Validation')
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                processed_data += inputs.size(0)
                
                pbar.set_postfix({
                    'loss': running_loss / processed_data,
                    'acc': running_corrects.double() / processed_data
                })
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_loader.dataset)
        
        return epoch_loss, epoch_acc

    def train(self):
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train phase
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch results
            print(f'Epoch {epoch}/{self.config["epochs"]}:')
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_wts,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                }, f'{self.config["checkpoint_dir"]}/best_model.pth')
            
            # Save checkpoint
            if epoch % self.config['save_freq'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': self.best_acc,
                }, f'{self.config["checkpoint_dir"]}/checkpoint_epoch_{epoch}.pth')
        
        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {self.best_acc:4f}')
        
        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model

if __name__ == '__main__':
    # Configuration for ImageNet-100
    config_imagenet100 = {
        'dataset': 'imagenet100',
        'data_dir': './imagenet/imagenet100',
        'batch_size': 256,
        'num_workers': 8,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 90,
        'pretrained': False,
        'checkpoint_dir': './checkpoints/imagenet100',
        'save_freq': 5
    }
    
    # Configuration for ImageNet-1000
    config_imagenet1000 = {
        'dataset': 'imagenet1000',
        'data_dir': './imagenet',
        'batch_size': 256,
        'num_workers': 8,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 90,
        'pretrained': False,
        'checkpoint_dir': './checkpoints/imagenet1000',
        'save_freq': 5
    }
    
    # Choose which configuration to use
    config = config_imagenet100  # or config_imagenet1000
    
    trainer = Trainer(config)
    model = trainer.train() 