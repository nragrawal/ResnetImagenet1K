import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import traceback
import sys
from network import create_model  # Import our model architecture

# Load model from local checkpoint
def load_model():
    try:
        model = create_model(num_classes=1000)  # Match your training classes
        checkpoint = torch.load('model/model_best.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        raise e

# Load ImageNet class labels
def load_labels():
    try:
        with open('model/classes.txt', 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except:
        return [f"Class_{i}" for i in range(100)]  # Fallback to generic labels

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Global variables
model = load_model()
labels = load_labels()

# Inference function
def predict(image):
    try:
        # Preprocess image
        img = Image.fromarray(image)
        img = transform(img).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(img)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        return {labels[idx]: float(prob) for prob, idx in zip(top5_prob, top5_catid)}
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

# Create Gradio interface with error handling
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="ResNet Image Classification",
    description="Upload an image to classify it using ResNet trained on ImageNet subset",
    allow_flagging="never"
)

# Add error handling to launch
try:
    iface.launch(share=True)
except Exception as e:
    print(f"Error launching interface: {str(e)}")
    print(traceback.format_exc()) 