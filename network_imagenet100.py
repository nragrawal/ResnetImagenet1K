import torch
import torch.nn as nn
from network import ResNet50

def create_model_imagenet100(pretrained=False):
    """
    Create ResNet50 model for ImageNet-100
    Args:
        pretrained: If True, initialize with ImageNet-1000 pretrained weights
                   and modify the final layer for 100 classes
    """
    model = ResNet50(num_classes=100)
    
    if pretrained:
        # Load pretrained weights from torchvision and modify the final layer
        import torchvision.models as models
        pretrained_model = models.resnet50(pretrained=True)
        # Remove the last layer weights
        pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() 
                          if 'fc' not in k}
        model.load_state_dict(pretrained_dict, strict=False)
        
        # Initialize the new final layer
        nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(model.fc.bias, 0)
    
    return model 