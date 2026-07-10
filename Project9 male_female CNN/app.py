
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Male/Female Image Classifier",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_gender_model():
    # Change the filename below if your model file has a different name
    model = load_model("model.keras")
    return model

model = load_gender_model()

# ----------------------------
# App title / intro
# ----------------------------
st.title("👤 Male / Female Image Classifier")
st.markdown(
    """
    Upload an image and the model will predict whether the face is **Male** or **Female**.
    """
)

# ----------------------------
# Class labels
# IMPORTANT:
# Adjust this based on how your model was trained.
# If train_generator.class_indices was:
# {'Female': 0, 'Male': 1}
# then keep this order exactly.
# ----------------------------
class_names = ["Female", "Male"]

# ----------------------------
# Image preprocessing function
# ----------------------------
def preprocess_image(image, target_size=(299, 299)):
    """
    Preprocess uploaded image for model prediction.
    Update target_size if your model was trained on a different size,
    e.g. (224,224), (128,128), etc.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape -> (1, H, W, 3)
    return img_array

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        try:
            processed_image = preprocess_image(image, target_size=(150, 150))
            prediction = model.predict(processed_image)

            # -----------------------------------
            # Handle binary output safely
            # Cases:
            # 1) shape (1,1) with sigmoid
            # 2) shape (1,2) with softmax
            # -----------------------------------
            if prediction.shape[-1] == 1:
                # sigmoid output
                confidence = float(prediction[0][0])

                if confidence >= 0.5:
                    predicted_label = class_names[1]   # Male
                    predicted_conf = confidence
                else:
                    predicted_label = class_names[0]   # Female
                    predicted_conf = 1 - confidence

            else:
                # softmax / multi-class style output
                predicted_index = int(np.argmax(prediction))
                predicted_label = class_names[predicted_index]
                predicted_conf = float(np.max(prediction))

            st.success(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence: **{predicted_conf * 100:.2f}%**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ----------------------------
# Sidebar instructions
# ----------------------------
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.write("1. Upload a JPG / JPEG / PNG image.")
    st.write("2. Click **Predict**.")
    st.write("3. The model will classify the image as Male or Female.")


