import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
@st.cache_resource()  # Ensures model loads only once for efficiency
def load_model():
    return tf.keras.models.load_model("DenseNet_model.h5")

model = load_model()

# Define class labels
class_labels = ["cloudy", "desert", "green_area", "water"]  # Update based on your dataset

# Streamlit UI
st.title("Satellite Image Classification 🌍")
st.write("Upload an image to classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64, 64))  # Match model input size
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Apply same preprocessing as training

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display results
    st.image(image, caption=f"Predicted: {predicted_class}", use_column_width=True)
    st.write(f"**Prediction:** {predicted_class} ✅")
