import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = load_model('mnist_cnn_model.h5')

# Define a function to preprocess the image for prediction
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (white background, black digits)
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255
    return image

# Streamlit UI
st.title("MNIST Digit Recognizer")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict using the loaded model
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction, axis=1)[0]

    # Show the result
    st.write(f"Predicted Digit: {predicted_digit}")
