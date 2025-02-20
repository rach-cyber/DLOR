import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load TensorFlow Lite model
@st.cache_resource()  # Ensures model loads only once
def load_model():
    interpreter = tf.lite.Interpreter(model_path="DenseNet_model (1).tflite")
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

# Get input and output details
input_details = model.get_input_details()
output_details = model.get_output_details()

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
    image_array = preprocess_input(image_array).astype(np.float32)  # Ensure correct dtype

    # Run inference using TensorFlow Lite
    model.set_tensor(input_details[0]['index'], image_array)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])

    # Get predicted class
    predicted_class = class_labels[np.argmax(prediction)]

    # Display results
    st.image(image, caption=f"Predicted: {predicted_class}", use_column_width=True)
    st.write(f"**Prediction:** {predicted_class} ✅")

