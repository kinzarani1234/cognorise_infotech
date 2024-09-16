import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Add some basic CSS styling with smaller sizes
st.markdown("""
    <style>
    .title {
        font-size: 35px;
        font-weight: bold;
        color: #3A3B3C;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #6C757D;
        text-align: center;
        margin-bottom: 0px;
    }
    .reportview-container {
        background-color: #f0f2f6;
    }
    .uploaded-image {
        max-width: 50%;  /* Ensure the image doesn't take up too much space */
        margin: auto;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 5px;
    }
    .file-uploader {
        display: flex;
        justify-content: center;
        margin: 0px 0;
        margin-bottom: 0px;
    }
    .file-uploader input[type="file"] {
        width: 300px;  /* Adjust the width as needed */
        height: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.markdown('<p class="title">COVID & Pneumonia Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an X-ray image to get a prediction</p>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_prediction_model():
    return load_model('best_model1.keras')

model = load_prediction_model()

# Initialize session state for tracking file upload and prediction status
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Image upload section
st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
st.markdown('</div>', unsafe_allow_html=True)

# Update session state with the uploaded file
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.prediction_made = False  # Reset prediction status when a new image is uploaded

# Show uploaded image and Predict button if an image has been uploaded
if st.session_state.uploaded_file is not None:
    image = Image.open(st.session_state.uploaded_file).convert('RGB')  # Ensure image is in RGB format
    st.image(image, caption='', width=150)  # Reduced image size

    # Predict button logic
    if st.button('Predict'):
        # Preprocess image
        image = image.resize((150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)

        # Display prediction
        if predicted_class == 0:
            st.write("Prediction: COVID Positive")
        elif predicted_class == 1:
            st.write("Prediction: Normal")
        else:
            st.write("Prediction: Pneumonia Positive")

        # Update prediction status
        st.session_state.prediction_made = True
        st.success("Prediction complete! Upload a new image to start again.")

# Display message if no image is uploaded
if st.session_state.uploaded_file is None:
    st.write("Please upload an image to get a prediction.")
