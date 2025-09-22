# MudraAI: AI-Based Recognition of Sattriya Mudras

MudraAI is an AI-powered image processing model designed to recognize 15 unique single-hand Sattriya mudras from images. This project leverages computer vision techniques and modern machine learning algorithms to provide accurate hand gesture recognition. The primary goal of this project is to combine traditional Indian dance with advanced AI techniques, preserving and promoting cultural heritage through technology.

## Project Overview

Sattriya is a classical dance form from Assam, India, that incorporates intricate hand gestures known as "Mudras". This project enables the recognition of 15 distinct single-hand Sattriya mudras, utilizing a deep learning model trained on a dataset of labeled hand gesture images. The recognition model is integrated with a responsive web application, providing users with an interactive experience.

### Features
- **Hand Gesture Recognition**: Detects 15 unique single-hand Sattriya mudras.
- **Deep Learning Model**: A neural network trained to classify mudras from images.
- **Web Interface**: Simple, responsive frontend built with HTML, CSS, JavaScript, and jQuery.
- **Backend**: Python-based Flask API handles image processing and gesture classification.

## Installation

### Prerequisites
Ensure you have Python 3.x and the following dependencies installed:
- Flask
- PyTorch
- OpenCV
- scikit-learn
- numpy
- pandas
- jQuery (for frontend interactivity)

You can install the necessary Python dependencies with the following command:

```bash
pip install -r requirements.txt
````

### Setting Up the Project

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/RakeshDas18/mudraai.git
   ```

2. Navigate to the project directory:

   ```bash
   cd mudra ai
   ```

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:

   ```bash
   python app.py
   ```

5. Open a web browser and visit `http://127.0.0.1:5000/` to start using the application.

## Usage

Once the app is running, users can upload images of their hand gestures. The system will process the image, classify the gesture using the trained deep learning model, and display the identified mudra along with a description of its meaning and significance in Sattriya dance.

### Example Workflow:

1. Open the app in your browser.
2. Upload an image of a hand gesture.
3. The model will analyze the image and identify the mudra.
4. The mudra name and description will be displayed on the screen.

## Model Training

The core model is a neural network trained on a dataset of labeled Sattriya mudra images. Here's a high-level overview of the training process:

1. **Data Collection**: Gathered a dataset of images representing 15 different Sattriya mudras.
2. **Preprocessing**: Images were resized, normalized, and converted into a suitable format for training.
3. **Model Architecture**: A convolutional neural network (CNN) was used for feature extraction and classification.
4. **Training**: The model was trained using PyTorch and saved as a `.pth` file (`mudra_model_with_unknown_3.pth`).
5. **Testing**: The model was evaluated on a test set for accuracy.

The trained model (`mudra_model_with_unknown_3.pth`) is loaded and used for real-time gesture classification in the Flask API.

## Frontend

The frontend is built using:

* **HTML**: Structure of the web page.
* **CSS**: Styling and layout.
* **JavaScript & jQuery**: For handling user interactions, like image uploads and displaying results.

The user interface is designed to be responsive, allowing users to access the application on both desktop and mobile devices.

## Contributing

We welcome contributions! If you'd like to improve the project, feel free to fork the repository and submit a pull request.

To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

## Acknowledgments

* **PyTorch**: For building and training the deep learning model.
* **OpenCV**: For computer vision tasks and image preprocessing.
* **Flask**: For backend API development.
* **Sattriya Dance**: For inspiring the project and helping preserve cultural heritage through technology.


This project is dedicated to preserving and promoting the traditional art of Sattriya through the power of technology and machine learning.
