import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Male / Female Image Classifier",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def get_model():
    return load_model("mod.keras")

model = get_model()

# ----------------------------
# Class labels
# Change this order if your training class order was different
# Example:
# ["Female", "Male"]  OR  ["Male", "Female"]
# ----------------------------
class_names = ["Female", "Male"]

# ----------------------------
# Image preprocessing
# Model input size = 299 x 299
# ----------------------------
def preprocess_image(image, target_size=(299, 299)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# App UI
# ----------------------------
st.title("👤 Male / Female Image Classifier")
st.write("Upload an image and the model will predict whether it is Male or Female.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        try:
            img = preprocess_image(image, target_size=(299, 299))
            pred = model.predict(img)

            # Case 1: sigmoid output (shape like [[0.82]])
            if pred.shape[-1] == 1:
                score = float(pred[0][0])

                if score >= 0.5:
                    label = class_names[1]
                    confidence = score
                else:
                    label = class_names[0]
                    confidence = 1 - score

            # Case 2: softmax output (shape like [[0.2, 0.8]])
            else:
                idx = int(np.argmax(pred))
                label = class_names[idx]
                confidence = float(np.max(pred))

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload a JPG / JPEG / PNG image.")
    st.write("2. Click Predict.")
    st.write("3. View the predicted result.")

    st.header("Model Details")
    st.write("Model file: `model.keras`")
    st.write("Input size: `299 x 299`")

    st.header("Important")
    st.write("If your model class order is different, update `class_names` in the code.")

