import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from PIL import Image

# Load the MNIST dataset from a CSV file
mnist_df = pd.read_csv('train.csv')

# Extracting features and labels
X = mnist_df.iloc[:, 1:].values
y = mnist_df.iloc[:, 0].values

# Reshape the data to (num_samples, 28, 28, 1) and normalize
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode the labels
y = to_categorical(y, 10)

# Build the CNN model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Streamlit app
st.title("Online Learning Model with Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    image = image.resize((28, 28))
    image_array = np.array(image.convert('L'))  # Convert to grayscale
    image_array = image_array.reshape((1, 28, 28, 1)).astype('float32') / 255.0

    # Make prediction
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)

    # Display predicted label
    st.write("Predicted Label:", predicted_label)

    # Allow user to input the true label
    true_label = st.number_input("Enter the true label (0-9):", min_value=0, max_value=9, step=1)

    # Fine-tune the model if the true label is provided
    if st.button("Submit True Label"):
        true_label_onehot = to_categorical(true_label, 10)
        model.fit(image_array, true_label_onehot, epochs=1, verbose=0)
        st.success("Model fine-tuned successfully!")

# Note: The button click will fine-tune the model with the provided true label.
# You may want to include additional error handling and user feedback.
