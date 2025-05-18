import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("digit_recognition_cnn.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit (0–9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    st.write(f"Predicted Digit: **{predicted_class}**"
