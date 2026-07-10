import streamlit as st
import numpy as np
import cv2
import joblib

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="👨🚺♀️",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------

model = joblib.load("male_female_model.pkl")

IMG_SIZE = 64

st.title("👨🚺♀️ Male vs Female Image Classifier")

st.write("Upload an image to predict whether the person is male  or female.")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, channels="BGR", width=300)

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    resized = resized.flatten()

    prediction = model.predict([resized])[0]

    probability = model.predict_proba([resized])[0]

    if prediction == 0:
        st.success("🧑‍🦱Prediction: Male")
    else:
        st.success("👧 Prediction: Female")

    st.write(f"Male Probability : {probability[0]*100:.2f}%")
    st.write(f"Female Probability : {probability[1]*100:.2f}%")