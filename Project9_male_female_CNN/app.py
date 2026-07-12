import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import time

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="VisionGender AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mod.keras")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# ----------------------------
# Labels
# ----------------------------
class_names = ["Female", "Male"]

# ----------------------------
# Preprocess image
# ----------------------------
def preprocess_image(image, target_size=(299, 299)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Session state
# ----------------------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
/* ========== GLOBAL ========== */
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(124,58,237,0.25), transparent 28%),
        radial-gradient(circle at top right, rgba(236,72,153,0.18), transparent 28%),
        radial-gradient(circle at bottom, rgba(59,130,246,0.15), transparent 30%),
        linear-gradient(135deg, #0b1020 0%, #111827 45%, #0f172a 100%);
    color: white;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}

/* ========== HERO ========== */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(124,58,237,0.22), rgba(236,72,153,0.16), rgba(59,130,246,0.18));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 28px;
    padding: 2rem 2rem 1.8rem 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
    animation: fadeIn 0.9s ease;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.hero:hover {
    transform: translateY(-4px);
    box-shadow: 0 28px 70px rgba(0,0,0,0.40);
}

.hero::before, .hero::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    filter: blur(55px);
    opacity: 0.55;
    z-index: 0;
}

.hero::before {
    width: 240px;
    height: 240px;
    top: -70px;
    right: -60px;
    background: rgba(236,72,153,0.28);
}

.hero::after {
    width: 260px;
    height: 260px;
    bottom: -90px;
    left: -70px;
    background: rgba(99,102,241,0.28);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-badge {
    display: inline-block;
    padding: 0.42rem 0.95rem;
    border-radius: 999px;
    background: linear-gradient(90deg, #7c3aed, #ec4899);
    color: white;
    font-weight: 700;
    font-size: 0.92rem;
    box-shadow: 0 10px 22px rgba(236,72,153,0.25);
    margin-bottom: 0.9rem;
}

.hero-title {
    font-size: 3rem;
    line-height: 1.15;
    font-weight: 900;
    margin: 0;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    margin-top: 0.85rem;
    font-size: 1.08rem;
    color: #dbeafe;
    max-width: 900px;
    line-height: 1.65;
}

/* ========== GLASS CARDS ========== */
.glass-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 1.25rem;
    box-shadow: 0 14px 40px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
    animation: slideUp 0.7s ease;
    transition: transform 0.28s ease, box-shadow 0.28s ease, border 0.28s ease;
}

.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 45px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.18);
}

.section-title {
    font-size: 1.22rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.8rem;
}

/* ========== METRIC CARDS ========== */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 1rem;
    text-align: center;
    min-height: 120px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    transition: transform 0.28s ease, box-shadow 0.28s ease, border 0.28s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.08), transparent);
    transform: translateX(-100%);
    transition: transform 0.6s ease;
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 18px 38px rgba(0,0,0,0.28);
    border: 1px solid rgba(255,255,255,0.18);
}

.metric-card:hover::before {
    transform: translateX(100%);
}

.metric-label {
    font-size: 0.95rem;
    color: #cbd5e1;
    margin-bottom: 0.45rem;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: white;
}

/* ========== FILE UPLOADER ========== */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1.5px dashed rgba(255,255,255,0.20);
    border-radius: 22px;
    padding: 1rem;
}

/* ========== BUTTON ========== */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;
    padding: 0.95rem 1rem;
    font-size: 1.02rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #7c3aed, #ec4899, #3b82f6);
    background-size: 200% 200%;
    box-shadow: 0 14px 28px rgba(124,58,237,0.32);
    transition: transform 0.22s ease, box-shadow 0.22s ease, filter 0.22s ease;
    animation: gradientMove 5s ease infinite;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 20px 36px rgba(236,72,153,0.35);
    filter: brightness(1.08);
}

.stButton > button:active {
    transform: scale(0.98);
}

/* ========== RESULT CARDS ========== */
.result-card {
    border-radius: 24px;
    padding: 1.4rem;
    margin-top: 1rem;
    box-shadow: 0 16px 40px rgba(0,0,0,0.28);
    animation: fadeIn 0.6s ease;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
}

.result-card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 22px 48px rgba(0,0,0,0.35);
}

.result-female {
    background: linear-gradient(135deg, rgba(236,72,153,0.22), rgba(244,114,182,0.16));
    border: 1px solid rgba(244,114,182,0.35);
}

.result-male {
    background: linear-gradient(135deg, rgba(59,130,246,0.22), rgba(99,102,241,0.16));
    border: 1px solid rgba(96,165,250,0.35);
}

.result-title {
    font-size: 1.6rem;
    font-weight: 900;
    margin-bottom: 0.5rem;
}

.result-text {
    font-size: 1.08rem;
    line-height: 1.7;
}

/* ========== CONFIDENCE RING ========== */
.ring-wrap {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

.ring {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    box-shadow: inset 0 0 25px rgba(255,255,255,0.05), 0 10px 30px rgba(0,0,0,0.25);
    transition: transform 0.28s ease, box-shadow 0.28s ease;
}

.ring:hover {
    transform: scale(1.03);
    box-shadow: inset 0 0 30px rgba(255,255,255,0.08), 0 16px 36px rgba(0,0,0,0.30);
}

.ring-inner {
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: rgba(15,23,42,0.95);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    border: 1px solid rgba(255,255,255,0.08);
}

.ring-value {
    font-size: 1.9rem;
    font-weight: 900;
    color: white;
}

.ring-label {
    font-size: 0.9rem;
    color: #cbd5e1;
}

/* ========== IMAGE PREVIEW HOVER ========== */
.image-hover-wrap {
    border-radius: 20px;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.image-hover-wrap:hover {
    transform: translateY(-6px);
    box-shadow: 0 18px 40px rgba(0,0,0,0.35);
}

/* ========== SIDEBAR ========== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 55%, #0b1220 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: transform 0.28s ease, box-shadow 0.28s ease, border 0.28s ease;
}

.sidebar-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.28);
    border: 1px solid rgba(255,255,255,0.16);
}

/* ========== FOOTER ========== */
.footer-box {
    margin-top: 1.5rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    text-align: center;
    color: #cbd5e1;
    transition: transform 0.28s ease, box-shadow 0.28s ease;
}

.footer-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
}

/* ========== ANIMATIONS ========== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Hero section
# ----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">✨ Premium AI Vision Interface</div>
        <h1 class="hero-title">VisionGender AI — Male / Female Eye Classifier</h1>
        <p class="hero-subtitle">
            Upload an image and get an AI-powered gender prediction with a polished dashboard, dynamic result cards,
            hover interactions, animated confidence visuals, and a more impressive project presentation.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ----------------------------
# Top metrics
# ----------------------------
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🧠 Model</div>
        <div class="metric-value">CNN / Keras</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📐 Input Size</div>
        <div class="metric-value">299 × 299</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📦 Model File</div>
        <div class="metric-value">mod.keras</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------------------
# Main layout
# ----------------------------
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload an Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a JPG / JPEG / PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.info("Drop an image here to start the prediction workflow.")

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.write("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖼 Uploaded Preview</div>', unsafe_allow_html=True)
        st.markdown('<div class="image-hover-wrap">', unsafe_allow_html=True)
        st.image(image, caption="Preview", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚀 App Highlights</div>', unsafe_allow_html=True)
    st.write("• Premium animated glassmorphism UI")
    st.write("• Hover effects across cards, hero, sidebar, and result panels")
    st.write("• Dynamic prediction reveal with confidence visualization")
    st.write("• Celebration effects after successful prediction")
    st.write("• Suitable for portfolio and internship project showcase")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧭 Prediction Flow</div>', unsafe_allow_html=True)
    st.write("1. Upload an eye image")
    st.write("2. Click **Predict Now**")
    st.write("3. Model processes the image")
    st.write("4. Result + confidence appear with visual feedback")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Prediction button
# ----------------------------
st.write("")
predict_col, spacer = st.columns([1.1, 0.9])

prediction_done = False
label = None
confidence = None

with predict_col:
    if uploaded_file is not None:
        if st.button("✨ Predict Now"):
            with st.spinner("Analyzing image and generating prediction..."):
                time.sleep(1.2)
                img = preprocess_image(image, target_size=(299, 299))
                pred = model.predict(img)

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

            st.session_state.last_prediction = label
            st.session_state.last_confidence = confidence
            prediction_done = True

# ----------------------------
# Result section
# ----------------------------
if prediction_done:
    st.balloons()

    is_female = (label == "Female")
    result_class = "result-female" if is_female else "result-male"
    emoji = "🌸" if is_female else "⚡"

    st.write("")
    res_col1, res_col2 = st.columns([1.1, 0.9], gap="large")

    with res_col1:
        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-title">{emoji} Prediction Complete</div>
                <div class="result-text">
                    <strong>Predicted Label:</strong> {label}<br>
                    <strong>Confidence:</strong> {confidence * 100:.2f}%<br><br>
                    The uploaded image was processed successfully by the trained model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.progress(min(max(float(confidence), 0.0), 1.0))
        st.success(f"Detected **{label}** with **{confidence * 100:.2f}%** confidence")

    with res_col2:
        pct = int(round(confidence * 100))
        ring_color = "#ec4899" if is_female else "#3b82f6"
        ring_bg = f"conic-gradient({ring_color} 0% {pct}%, rgba(255,255,255,0.10) {pct}% 100%)"

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">📊 Confidence Meter</div>
                <div class="ring-wrap">
                    <div class="ring" style="background:{ring_bg};">
                        <div class="ring-inner">
                            <div class="ring-value">{pct}%</div>
                            <div class="ring-label">Confidence</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🌟 VisionGender AI")

    st.markdown("""
    <div class="sidebar-card">
        <h4>📌 Instructions</h4>
        <p>• Upload a JPG / JPEG / PNG image<br>
        • Click <strong>Predict Now</strong><br>
        • Review the predicted label and confidence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>🧠 Model Details</h4>
        <p><strong>Framework:</strong> TensorFlow / Keras<br>
        <strong>Input Size:</strong> 299 × 299<br>
        <strong>Model File:</strong> mod.keras</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>🎨 UI Features</h4>
        <p>• Cinematic hero section<br>
        • Hover effects on cards and result panels<br>
        • Dynamic confidence ring<br>
        • Celebration effects<br>
        • Premium glassmorphism layout</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_prediction is not None:
        st.markdown(f"""
        <div class="sidebar-card">
            <h4>🕘 Last Prediction</h4>
            <p><strong>Label:</strong> {st.session_state.last_prediction}<br>
            <strong>Confidence:</strong> {st.session_state.last_confidence * 100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>⚠️ Class Order</h4>
        <p>If your training labels were different, update:</p>
    </div>
    """, unsafe_allow_html=True)

    st.code('class_names = ["Female", "Male"]', language="python")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<div class="footer-box">
    Built with Streamlit, TensorFlow, and a custom premium UI layer for a polished internship / portfolio demo.
</div>
""", unsafe_allow_html=True)
