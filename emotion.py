import streamlit as st
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom CNN model architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # 1st Convolution Layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        # 2nd Convolution Layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        # 3rd Convolution Layer
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        # 4th Convolution Layer
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.25)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 3 * 3, 256)  # Adjust based on input size
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(256, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.drop_fc2 = nn.Dropout(0.25)

        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        # Convolution + Activation + Pooling + Dropout
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)

        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop4(x)

        # Flatten the output
        x = torch.flatten(x, start_dim=1)

        # Fully Connected Layers
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))

        # Output layer
        x = self.fc3(x)
        return x

# Load the CustomCNN model
@st.cache_resource
def load_model():
    model = CustomCNN().to(device)
    model.load_state_dict(torch.load(r"C:\Users\peer1\OneDrive\Documents\Guvi_Projects\data\custom_emotion_model.pth", map_location=device))
    model.eval()
    return model

# Define the transformations
def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

# Class names corresponding to the emotion labels
class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Prediction function
def predict_emotion(image, model, transform):
    # Process the image
    processed_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    model = model.to(device)  # Ensure the model is on the same device
    with torch.no_grad():
        output = model(processed_image)
        _, predicted = torch.max(output, 1)
        emotion = class_names[predicted.item()]
    return emotion

# Streamlit UI
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    initial_sidebar_state="expanded"
)
st.markdown("<h1>🎭 Emotion Detection App</h1>", unsafe_allow_html=True)

# Load the model and transformations
model = load_model()
transform = get_transform()

# Image upload

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("File size exceeds 5MB. Please upload a smaller file.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)  

        # Perform emotion detection
        with st.spinner("Analyzing the image..."):
            emotion = predict_emotion(image, model, transform)

        # Display results
        st.write(f"**Predicted Emotion:** {emotion}")

