from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torchvision import models
import torch.nn as nn
import sklearn

app = Flask(__name__)

# Load the trained SVM model
svm_classifier = joblib.load('C:/Users/tonyn/math457/svm_model2.pkl')

# Load the pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Define the transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def extract_features_from_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Move the image tensor to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    with torch.no_grad():
        features = resnet(image)

    # Convert features to a numpy array
    features = features.squeeze().cpu().numpy()

    return features


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file from the form
        uploaded_file = request.files['file']

        # Save the file temporarily
        temp_path = 'temp_image.jpg'
        uploaded_file.save(temp_path)

        # Extract features from the uploaded image
        image_features = extract_features_from_image(temp_path)

        # Make a prediction using the SVM model
        prediction_label = svm_classifier.predict(image_features.reshape(1, -1))[0]

        # Get probabilities
        probabilities = svm_classifier.predict_proba(image_features.reshape(1, -1))[0]
        probability_cat = probabilities[0] * 100
        probability_dog = probabilities[1] * 100

        # Return the prediction result
        return render_template('result.html',
                               prediction_label=prediction_label,
                               probability_cat=probability_cat,
                               probability_dog=probability_dog)

    except Exception as e:
        return render_template('error.html', error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True)
