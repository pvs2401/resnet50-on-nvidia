import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import json
import os
import time

MODEL_NAME = "resnet50"
DATASET_NAME = "food101"
DATASET_LABEL_MAPPING_FILE = "./results/food101_label_mapping.json"
MODEL_STATE = "./results/models/resnet50_best_model_state.pth"
DIAGRAM_PATH = './results/plots/results.png'  # Path to the diagram

# Model Class
class ResNetModelClass(nn.Module):
    def __init__(self, num_classes=101):  # Default to 101 classes for Food-101
        super(ResNetModelClass, self).__init__()
        self.resnet50 = models.resnet50()  # Load ResNet-50 without its pre-trained parameters
        self.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)  # Define final layer explicitly
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)  # Change final layer
        self.model_name = MODEL_NAME  # Add model name as an attribute
        
    def forward(self, x):
        return self.resnet50(x)

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load model from class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetModelClass()

# Load the final save parameters from the trained model (state_dict) into the model
state_dict = torch.load(MODEL_STATE, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.eval()  # Set model to evaluation mode

# Enable DataParallel for multi-GPU inference
if torch.cuda.device_count() > 1:
    #print(f"[INFO] Using {torch.cuda.device_count()} GPUs for inference")
    model = nn.DataParallel(model)
model = model.to(device)

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a list of all class names
with open(DATASET_LABEL_MAPPING_FILE, "r") as f:
    label_mapping = json.load(f)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    START_TIME = time.time()
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output[0], dim=0)
        predicted_class_index = probabilities.argmax().item()
        confidence = probabilities[predicted_class_index].item()

        # Get the class name from the index
        predicted_class_name = label_mapping[predicted_class_index]
        # Identify GPU used for computation (using CUDA's current device)
        #gpu_id = input_tensor.device.index if input_tensor.device.type == "cuda" else "CPU"
        #print(f"GPU ID is: {gpu_id}")
    DURATION = time.time() - START_TIME
    return jsonify({
        'predicted_class_id': predicted_class_index,
        'predicted_class_name': predicted_class_name,
        'confidence': confidence,
        'duration': DURATION
    })

# Route to serve the diagram
@app.route('/diagram', methods=['GET'])
def get_diagram():
    # Serve the diagram image from the file system
    if os.path.exists(DIAGRAM_PATH):
        return send_from_directory(os.path.dirname(DIAGRAM_PATH), os.path.basename(DIAGRAM_PATH))
    else:
        return jsonify({'error': 'Diagram not found'}), 404

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Starting Inference using Model: {MODEL_NAME} using saved model parameters from File: {MODEL_STATE}")
    app.run(host='0.0.0.0', port=9999)