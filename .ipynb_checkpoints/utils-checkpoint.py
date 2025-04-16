import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Accuracy calculation
def calculate_accuracy(y_pred, y_true):
    _, preds = torch.max(y_pred, 1)
    correct = (preds == y_true).sum().item()
    return correct / len(y_true)

# Show image and prediction
def show_image(img_tensor, label, predicted=None):
    img = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img)
    if predicted:
        plt.title(f"True: {label} | Predicted: {predicted}")
    else:
        plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# Encode breed names
def encode_labels(breed_list):
    le = LabelEncoder()
    encoded = le.fit_transform(breed_list)
    return encoded, le

# Load single image and preprocess for prediction
def load_image_for_prediction(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)
