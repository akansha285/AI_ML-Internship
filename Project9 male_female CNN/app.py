import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
page_title="Eye Gender Detection",
page_icon="👁️",
layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""

<style>
.main {
    background-color: #f7f9fc;
}

.title {
    font-size: 40px;
    font-weight: 800;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #4a4a4a;
    text-align: center;
    margin-bottom: 25px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.result-box {
    background: linear-gradient(135deg, #dff6ff, #ffffff);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: #0b5394;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.footer-text {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #1f77b4, #4facfe);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1565c0, #2196f3);
    color: white;
}
</style>

""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown('<div class="title">👁️ Eye Gender Detection App</div>', unsafe_allow_html=True)
st.markdown(
'<div class="subtitle">Upload an eye image and the model will predict whether it belongs to a <b>Male</b> or <b>Female</b>.</div>',
unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_my_model():
return load_model("model.keras")

model = load_my_model()

# ---------------- IMAGE PREPROCESS ----------------

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((299, 299))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader("📤 Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.image(img, caption="Uploaded Eye Image", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔍 Predict Gender"):
    with st.spinner("Analyzing image..."):
        img_array = preprocess_image(img)
        result = model.predict(img_array)
        score = float(result[0][0])

        if score > 0.5:
            prediction = "Male 👦"
            confidence = score * 100
        else:
            prediction = "Female 👧"
            confidence = (1 - score) * 100

    st.markdown(
        f"""
        <div class="result-box">
            Prediction: {prediction}<br>
            Confidence: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- FOOTER ----------------

st.markdown(
'<div class="footer-text">Built with ❤️ using Streamlit & Keras</div>',
unsafe_allow_html=True
)
