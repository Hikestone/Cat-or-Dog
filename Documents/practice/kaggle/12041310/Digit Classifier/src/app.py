import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from model import DigitClassifier, MNISTDataset
from torch.utils.data import DataLoader, Dataset

# Define transformations for image preprocessing
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)

# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load("../model/digit_classifier.pth"))
model.eval()

# Create directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

# Create a DataFrame or load existing CSV file
csv_file_path = "../data/pixel_values.csv"
columns = ["label"] + [f"pixel{i}" for i in range(784)]
if os.path.exists(csv_file_path):
    corrections_df = pd.read_csv(csv_file_path)
else:
    corrections_df = pd.DataFrame(columns=columns)

# Function to get pixel values from the image
def get_pixel_values(image_path):
    image = Image.open(image_path)
    pixel_values = np.array(image).flatten()
    return pixel_values

# Function to make predictions
def predict(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)

    predicted_class = output.argmax(dim=1, keepdim=True)
    return predicted_class.item()

def empty_csv_file(file_path):
    # Read the header of the CSV file
    with open(file_path, 'r') as file:
        header = file.readline()

    # Open the CSV file in write mode and write the header
    with open(file_path, 'w') as file:
        file.write(header)

# Streamlit app
st.title("Digit Classifier")
image = Image.open("../static/hero.png")
st.image(image, caption="")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def fine_tune(model, train_loader, num_epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.type(torch.LongTensor))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1, keepdim=True)
            # print(predicted)
            correct_predictions += predicted.eq(labels.view_as(predicted)).sum().item()
            # print(correct_predictions)
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Accuracy: {epoch_accuracy}")
    # print(total_samples)
    torch.save(model.state_dict(), "../model/digit_classifier.pth")


if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform prediction
    class_index = predict(uploaded_file)

    st.write(f"Prediction: {class_index}")
    lbl_txt = ""
    correct_label = st.text_input("Enter Correct Label:", lbl_txt)
    if correct_label != "":
        pixel_values = get_pixel_values(uploaded_file)
        row_data = [int(correct_label)] + pixel_values.tolist()
        new_row_df = pd.DataFrame([row_data], columns=columns)
        corrections_df = pd.concat([corrections_df, new_row_df], ignore_index=True)
        corrections_df.to_csv(csv_file_path, index=False)
    # print(corrections_df)

if st.button("Fine-tune Model"):
    corrections_df = pd.read_csv(csv_file_path)
    if not corrections_df.empty:
        train_images = MNISTDataset(corrections_df, transform=transform)
        train_loader = DataLoader(train_images, batch_size=32, shuffle=True)
        fine_tune(model, train_loader, num_epochs=10, lr=0.001)
        empty_csv_file(csv_file_path)
        corrections_df = pd.DataFrame(columns=columns)
        st.write("Model fine-tuned with new data!")
    else:
        st.write("No data for fine-tuning. Please upload images and provide correct labels.")

