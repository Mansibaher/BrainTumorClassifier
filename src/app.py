import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Grad-CAM function
def generate_gradcam(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor.requires_grad_()

    final_conv = model.layer4[-1].conv2  # Last conv layer
    grad = []
    fmap = []

    def save_grad(module, input, output):
        fmap.append(output)
        output.register_hook(lambda g: grad.append(g))

    handle = final_conv.register_forward_hook(save_grad)
    _ = model(image_tensor)

    handle.remove()

    fmap = fmap[0].squeeze().detach().numpy()
    grad = grad[0].squeeze().detach().numpy()
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:])

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

# UI
st.title("🧠 Brain Tumor MRI Classifier with Explainable AI")
st.write("Upload an MRI image and get tumor prediction with Grad-CAM heatmap.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image)
    outputs = model(input_tensor.unsqueeze(0))
    _, predicted = torch.max(outputs, 1)
    label = "🧠 Tumor Detected" if predicted.item() == 1 else "✅ No Tumor Detected"

    st.subheader(f"Prediction: {label}")

    # Grad-CAM
    cam = generate_gradcam(model, input_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)

    st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)
