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


# UNKNOWN_THRESHOLD = 0.7  

def predict_mudra(image_path):
    try:
        image = preprocess_image(image_path)
        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            max_prob, predicted_class = torch.max(probabilities, 1)

            # if max_prob.item() < UNKNOWN_THRESHOLD:
            #     return 'Unknown Mudra'
            # else:
            print(max_prob, predicted_class)
            return Categories[predicted_class.item()]
    except Exception as e:
        return f"Prediction error: {str(e)}"



def mudra_describe(mudra):
    descrption = {  'alopodmo':"Alapadma (Alopodmo) - A fully bloomed lotus. The fingers are spread out and curved slightly, symbolizing beauty, grace, or offering of flowers",
                    'ankush': "Ankusha (Ankush) – Represents an elephant goad or control. Often symbolizes discipline or steering.", 
                    'ardhachandra': "Ardhachandra – A half-moon shape. The thumb is extended while other fingers are together and straight. Used to represent the moon, a platter, or even to bless.", 
                    'bhramar': "Bhramara – The thumb and middle finger touch while the other fingers are curved. It depicts a bee and is also used to represent Krishna, his ornaments, or various animals.", 
                    'chatur': "Chatura (Chatur) – The thumb touches the ring finger, while the other fingers remain slightly bent. It can show a clever or graceful act, or even a bird.", 
                    'ghronik': "Ghronika (Ghronik) – Typically used to denote sniffing or smelling. Not very common in all classical dances but has expressive value in storytelling.", 
                    'hongshashyo': "Hansasya (Hongshashyo) – A swan-like hand; thumb and index finger touch gently while other fingers are extended. Used to denote lightness, grace, or delicate actions.", 
                    'kangul': "Kangula (Kangul) – Used to show a bell or anklet. The hand shape is stylized and decorative, suitable for symbolic representation.", 
                    'kodombo': "Kundamva (Kodombo) – A specific gesture, less common, possibly derived from local interpretations. Sometimes used for flowers or buds.", 
                    'kopitho': "Kapittha (Kopitho) – The thumb is pressed against the bent index finger, while other fingers are closed. It represents Lakshmi or Saraswati, or holding a cymbal", 
                    'krishnaxarmukh': "Krishna’s Face (Krishnaxarmukh) – A symbolic mudra representing the face of Lord Krishna, often used in devotional storytelling scenes.", 
                    'mrigoshirsho': "Mriga Shirsha (Mrigoshirsho) – The tips of the thumb and middle finger touch; others are extended. Represents a deer’s head or face, also used to depict searching or gentle animals.", 
                    'mukul': "Mukul – All fingers brought together to a point, like a lotus bud. Used for offering, showing flowers, or fruits." , 
                    'unknown': "Please Upload a Single Hand Sattriya Dance Mudra"
                }
    return descrption[mudra]


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
        result = mudra_describe(prediction)
        return jsonify({'prediction': result})

# **Main Route**
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)