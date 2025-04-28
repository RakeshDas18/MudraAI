from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define SVM model
class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten
        return self.fc(x)

# Define categories
Categories = ['alopodmo', 'ankush', 'ardhachandra', 'bhramar', 'chatur', 'ghronik', 'hongshashyo', 'kangul', 'kodombo', 'kopitho', 
              'krishnaxarmukh', 'mrigoshirsho', 'mukul', 'unknown']

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 150 * 150 * 3
num_classes = len(Categories)

model = SVM(input_dim, num_classes).to(device)
model.load_state_dict(torch.load('mudra_model_7_with_unknown.pth', map_location=device)) 
model.eval()  


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  
    return image


UNKNOWN_THRESHOLD = 0.7  

def predict_mudra(image_path):
    image = preprocess_image(image_path)
    
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities = torch.softmax(output, dim=1)  
        max_prob, predicted_class = torch.max(probabilities, 1)


        if max_prob.item() < UNKNOWN_THRESHOLD:
            un = 'Unknown Mudra'
            return un
        else:
            print(max_prob.item())
            print(f"Predicted Mudra: {Categories[predicted_class.item()]}")
            return Categories[predicted_class]



# **Fix: Add an upload route**
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict Mudra
        prediction = predict_mudra(filepath)

        return jsonify({'prediction': prediction})

# **Main Route**
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)