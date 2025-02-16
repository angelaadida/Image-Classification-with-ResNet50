import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load pre-trained model
model = ResNet50(weights="imagenet")

st.title("Image Classification with ResNet50")
st.write("Upload an image, and the app will classify it using a pretrained ResNet50 model.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((224, 224))  # Fix resizing syntax
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)
    decoded_predictions = decode_predictions(prediction, top=3)[0]  # Fix variable name

    # Display predictions
    st.write("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}. {label}: {score:.4f}")
