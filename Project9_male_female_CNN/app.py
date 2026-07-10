import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import time

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mod.keras")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Male / Female Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #111827);
    color: white;
}

/* Remove default top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(236,72,153,0.20));
    border: 1px solid rgba(255,255,255,0.12);
    padding: 2rem;
    border-radius: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
    animation: fadeIn 0.8s ease-in-out;
}

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    backdrop-filter: blur(8px);
    animation: slideUp 0.7s ease-in-out;
}

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.22), rgba(16,185,129,0.18));
    border: 1px solid rgba(74,222,128,0.35);
    border-radius: 20px;
    padding: 1.4rem;
    margin-top: 1rem;
    box-shadow: 0 10px 24px rgba(16,185,129,0.18);
}

/* Danger / info badge */
.badge {
    display: inline-block;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    color: white;
    box-shadow: 0 6px 18px rgba(236,72,153,0.28);
}

/* Big title */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    line-height: 1.2;
    color: white;
}

/* Subtitle */
.subtitle {
    font-size: 1.05rem;
    color: #dbeafe;
    opacity: 0.95;
}

/* Section heading */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    color: #f8fafc;
}

/* Sidebar style */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1f2937);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Custom button */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    font-size: 1rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #7c3aed, #ec4899);
    box-shadow: 0 8px 20px rgba(124,58,237,0.35);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 12px 24px rgba(236,72,153,0.35);
}

/* File uploader box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.22);
    border-radius: 18px;
    padding: 1rem;
}

/* Metric box */
.metric-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(12px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes slideUp {
    from {opacity: 0; transform: translateY(18px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# ----------------------------
# Class labels
# ----------------------------
class_names = ["Female", "Male"]

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(image, target_size=(299, 299)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Hero Section
# ----------------------------
st.markdown("""
<div class="hero-card">
    <div class="badge">AI Vision Demo</div>
    <div class="main-title">👤 Male / Female Image Classifier</div>
    <div class="subtitle">
        Upload an image and get a clean AI-based gender prediction with confidence score, polished visuals, and interactive feedback.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ----------------------------
# Layout
# ----------------------------
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a JPG / JPEG / PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.info("Upload an image to start prediction.")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.write("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖼 Uploaded Preview</div>', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Quick Info</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <h3 style="margin:0;">Model Input</h3>
            <p style="font-size:1.1rem; margin-top:8px;">299 × 299</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-box">
            <h3 style="margin:0;">Model File</h3>
            <p style="font-size:1.1rem; margin-top:8px;">mod.keras</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="metric-box">
        <h3 style="margin:0;">Prediction Labels</h3>
        <p style="font-size:1rem; margin-top:8px;">Female / Male</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Prediction Section
# ----------------------------
st.write("")
predict_col1, predict_col2 = st.columns([1.15, 0.85], gap="large")

with predict_col1:
    if uploaded_file is not None:
        if st.button("✨ Predict Now"):
            try:
                with st.spinner("Analyzing image and generating prediction..."):
                    time.sleep(1.2)  # small motion-like wait effect
                    img = preprocess_image(image, target_size=(299, 299))
                    pred = model.predict(img)

                # Binary output
                if pred.shape[-1] == 1:
                    score = float(pred[0][0])

                    if score >= 0.5:
                        label = class_names[1]
                        confidence = score
                    else:
                        label = class_names[0]
                        confidence = 1 - score
                else:
                    idx = int(np.argmax(pred))
                    label = class_names[idx]
                    confidence = float(np.max(pred))

                # Celebration effect
                st.balloons()

                st.markdown(f"""
                <div class="result-card">
                    <h2 style="margin-bottom:0.5rem;">🎉 Prediction Complete</h2>
                    <p style="font-size:1.15rem; margin:0.3rem 0;"><strong>Predicted Label:</strong> {label}</p>
                    <p style="font-size:1.1rem; margin:0.3rem 0;"><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.progress(min(max(float(confidence), 0.0), 1.0))
                st.success(f"Result: {label} detected with {confidence * 100:.2f}% confidence")

            except Exception as e:
                st.error(f"Prediction error: {e}")

with predict_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 How it works</div>', unsafe_allow_html=True)
    st.write("1. Upload a face image in JPG / JPEG / PNG format.")
    st.write("2. Click **Predict Now**.")
    st.write("3. The app preprocesses the image to **299 × 299**.")
    st.write("4. The trained CNN model predicts the class.")
    st.write("5. Confidence score and result are displayed with celebration effects.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🌟 App Dashboard")
    st.markdown("A polished AI demo for image-based classification.")

    st.markdown("---")
    st.markdown("### 📌 Instructions")
    st.write("• Upload a JPG / JPEG / PNG image")
    st.write("• Click **Predict Now**")
    st.write("• Review the prediction and confidence")

    st.markdown("---")
    st.markdown("### 🧠 Model Details")
    st.write("**Model file:** `mod.keras`")
    st.write("**Input size:** `299 x 299`")
    st.write("**Framework:** TensorFlow / Keras")

    st.markdown("---")
    st.markdown("### ⚠️ Important")
    st.write("If your training label order is different, update:")
    st.code('class_names = ["Female", "Male"]', language="python")

    st.markdown("---")
    st.markdown("### ✨ UI Features")
    st.write("• Gradient hero section")
    st.write("• Glassmorphism cards")
    st.write("• Animated buttons")
    st.write("• Celebration balloons")
    st.write("• Confidence progress bar")
