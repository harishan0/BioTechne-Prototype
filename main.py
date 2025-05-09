import streamlit as st
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Configuration
MODEL_PATH = "defect_resnet50_finetuned.pth"  # Path to your downloaded model weights
CLASS_NAMES = ['good', 'defect']

# Model definition (must match training architecture)
class DefectDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet50(pretrained=False)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.base_model(x)

# Load model
def load_model():
    model = DefectDetector()
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.base_model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# Image preprocessing
def process_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
st.set_page_config(page_title="Component Inspector", layout="wide")
st.title("🔍 Defect Detection App")

# Sidebar: image upload and threshold
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a component image", type=['png', 'jpg', 'jpeg'])
threshold = st.sidebar.slider("Defect Threshold", 0.0, 1.0, 0.5)

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Uploaded Image Preview")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
    else:
        st.info("Upload an image to analyze.")

with col2:
    st.subheader("Analysis Results")
    if uploaded_file:
        # Process image and predict
        tensor = process_image(image)
        with torch.no_grad():
            output = model(tensor)
            good_prob = torch.sigmoid(output)[0].item()
            defect_prob = 1.0 - good_prob

        # Display result metric
        st.metric("Defect Probability", f"{defect_prob:.2%}")

        # Classification label
        if defect_prob > threshold:
            st.error("Defect Detected!")
        else:
            st.success("Component OK")

        # Bar chart of confidence
        fig, ax = plt.subplots()
        ax.barh(['Good', 'Defect'], [good_prob, defect_prob])
        ax.set_xlim(0, 1)
        st.pyplot(fig)
    else:
        st.info("No image to analyze.")

st.markdown("---")
st.caption("Model: ResNet50 fine-tuned for defect detection. Upload custom images or integrate into your pipeline.")
