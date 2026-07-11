import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(
    page_title="Canada Income Predictor",
    page_icon="🇨🇦",
    layout="wide"
)

# -------------------------------------
# CUSTOM CSS
# -------------------------------------
st.markdown("""
<style>

.stApp{
    background:#f4f8fb;
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
    padding:25px;
    border-radius:20px;
    background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    box-shadow:0px 8px 20px rgba(0,0,0,0.25);
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:20px;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.stButton>button{
    width:100%;
    height:60px;
    font-size:22px;
    font-weight:bold;
    border-radius:12px;
    border:none;
    color:white;
    background:linear-gradient(90deg,#11998e,#38ef7d);
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.03);
    box-shadow:0px 8px 20px rgba(0,0,0,0.3);
}

.prediction{
    background:linear-gradient(90deg,#00b09b,#96c93d);
    color:white;
    padding:25px;
    border-radius:18px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    box-shadow:0px 8px 20px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------
# HEADER
# -------------------------------------
st.markdown(
    "<div class='title'>🇨🇦 Canada Per Capita Income Predictor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Predict Canada's Per Capita Income using Linear Regression</p>",
    unsafe_allow_html=True
)

st.write("")

# -------------------------------------
# LOAD DATA
# -------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "canada_per_capita_income.csv")

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

df = load_data()

# -------------------------------------
# SHOW DATA
# -------------------------------------
with st.expander("📄 View Dataset"):
    st.dataframe(df, use_container_width=True)

# -------------------------------------
# TRAIN MODEL
# -------------------------------------
X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

# -------------------------------------
# CHART
# -------------------------------------
st.subheader("📈 Historical Data")

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(
    df["year"],
    df["per capita income (US$)"],
    marker="o"
)

ax.plot(
    df["year"],
    model.predict(X)
)

ax.set_xlabel("Year")
ax.set_ylabel("Per Capita Income (US$)")
ax.grid(True)

st.pyplot(fig)

# -------------------------------------
# PREDICTION
# -------------------------------------
st.subheader("🔮 Predict Future Income")

year = st.slider(
    "Select Year",
    1970,
    2050,
    2025
)

if st.button("Predict Income"):

    prediction = model.predict([[year]])[0]

    st.balloons()

    st.markdown(
        f"""
        <div class='prediction'>
        💰 Estimated Per Capita Income<br><br>
        ${prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------
# MODEL DETAILS
# -------------------------------------
st.write("")
st.subheader("📊 Model Statistics")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Coefficient",
        f"{model.coef_[0]:.2f}"
    )

with col2:
    st.metric(
        "Intercept",
        f"{model.intercept_:.2f}"
    )

# -------------------------------------
# FOOTER
# -------------------------------------
st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit | Machine Learning | Linear Regression"
)
