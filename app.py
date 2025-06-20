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
Categories = ['Alopodmo', 'Ankush', 'Ardhachandra', 'Bhramar', 'Chatur', 'Ghronik', 'Hongshashyo', 'Kangul', 'Kodombo', 'Kopitho', 'Krishnaxarmukh', 'Mrigoshirsho', 'Mukul', 'Unknown']

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 256 * 256 * 3
num_classes = len(Categories)

model = SVM(input_dim, num_classes).to(device)
model.load_state_dict(torch.load('mudra_model_8_2.pth', map_location=device)) 
model.eval()  


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  
    return image


UNKNOWN_THRESHOLD = 0.7  

def predict_mudra(image_path):
    try:
        image = preprocess_image(image_path)
        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            max_prob, predicted_class = torch.max(probabilities, 1)

            if max_prob.item() < UNKNOWN_THRESHOLD:
                return 'unknown'
            else:
                print(max_prob, predicted_class)
                return Categories[predicted_class.item()]
    except Exception as e:
        return f"Prediction error: {str(e)}"



def mudra_describe(mudra):
    descrption = {  'Alopodmo':"Alapadma (Alopodmo) - A fully bloomed lotus. The fingers are spread out and curved slightly, symbolizing beauty, grace, or offering of flowers",
                    'Ankush': "Ankusha (Ankush) – Represents an elephant goad or control. Often symbolizes discipline or steering.", 
                    'Ardhachandra': "Ardhachandra – A half-moon shape. The thumb is extended while other fingers are together and straight. Used to represent the moon, a platter, or even to bless.", 
                    'Bhramar': "Bhramara – The thumb and middle finger touch while the other fingers are curved. It depicts a bee and is also used to represent Krishna, his ornaments, or various animals.", 
                    'Chatur': "Chatura (Chatur) – The thumb touches the ring finger, while the other fingers remain slightly bent. It can show a clever or graceful act, or even a bird.", 
                    'Ghronik': "Ghronika (Ghronik) – Typically used to denote sniffing or smelling. Not very common in all classical dances but has expressive value in storytelling.", 
                    'Hongshashyo': "Hansasya (Hongshashyo) – A swan-like hand; thumb and index finger touch gently while other fingers are extended. Used to denote lightness, grace, or delicate actions.", 
                    'Kangul': "Kangula (Kangul) – Used to show a bell or anklet. The hand shape is stylized and decorative, suitable for symbolic representation.", 
                    'Kodombo': "Kundamva (Kodombo) – A specific gesture, less common, possibly derived from local interpretations. Sometimes used for flowers or buds.", 
                    'Kopitho': "Kapittha (Kopitho) – The thumb is pressed against the bent index finger, while other fingers are closed. It represents Lakshmi or Saraswati, or holding a cymbal", 
                    'Krishnaxarmukh': "Krishna’s Face (Krishnaxarmukh) – A symbolic mudra representing the face of Lord Krishna, often used in devotional storytelling scenes.", 
                    'Mrigoshirsho': "Mriga Shirsha (Mrigoshirsho) – The tips of the thumb and middle finger touch; others are extended. Represents a deer’s head or face, also used to depict searching or gentle animals.", 
                    'Mukul': "Mukul – All fingers brought together to a point, like a lotus bud. Used for offering, showing flowers, or fruits." , 
                    'Unknown': "Please Upload a Single Hand Sattriya Dance Mudra"
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
        return jsonify({'prediction': result,
                        'mudra': prediction})

# **Main Route**
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/popup')
def popup():
    info = request.args.get('info', '')
    mudra = request.args.get('mudra')
    
    mudra_map = {
                    'Alopodmo': 'alopodmo.jpg',
                    'Ankush': 'ankush.jpg',
                    'Ardhachandra': 'ardhachandra.jpg',
                    'Bhramar': 'bhramar.jpg',
                    'Chatur': 'chatur.png',
                    'Ghronik': 'ghronik.jpg',
                    'Hongshashyo': 'hongshashyo.jpg',
                    'Kangul': 'kangul.png',
                    'Kodombo': 'kodombo.jpg',
                    'Kopitho': 'kopitho.jpg',
                    'Mrigoshirsho': 'mrigoshirsho.png',
                    'Krishnaxarmukh': 'krishnaxarmukh.png',
                    'Mukul': 'mukul.png',
                    'Unknown': ''
                }

    mudra_file = mudra_map.get(mudra, None)  # Handle unknown mudra gracefully

    return render_template('popup.html', info=info, mudra=mudra, mudra_file=mudra_file)


if __name__ == '__main__':
    app.run(debug=True)