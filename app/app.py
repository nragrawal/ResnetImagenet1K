import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from network import create_model

# Load model
def load_model():
    model = create_model(num_classes=1000)  # Adjust number of classes
    model.load_state_dict(torch.load('model/model_best.pth', map_location='cpu'))
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Inference function
def predict(image):
    model = load_model()
    
    # Preprocess image
    img = Image.fromarray(image)
    img = transform(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    return {f"Class {i}": float(prob) for i, prob in zip(top5_catid, top5_prob)}

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="ResNet Image Classification",
    description="Upload an image to classify it using ResNet"
)

iface.launch() 