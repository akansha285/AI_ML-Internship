import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Life Insurance Buying Prediction",
    page_icon="💸💰",
    layout="centered"
)

st.title("💸💰 Life Insurance Prediction")
st.write("Predict whether a person will buy life insurance based on age using Logistic Regression")

# Load dataset
import os
import pandas as pd

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to the CSV file
CSV_PATH = os.path.join(BASE_DIR, "insurance_data.csv")

# Read the CSV
df = pd.read_csv(CSV_PATH)

st.subheader("Dataset")
st.dataframe(df)

# Check required columns
required_columns = ['age', 'bought_insurance']
if not all(col in df.columns for col in required_columns):
    st.error("CSV file must contain 'age' and 'bought_insurance' columns.")
    st.stop()

# Features and target
X = df[['age']]
y = df['bought_insurance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# User input
st.subheader("Enter Person's Age")
age = st.number_input(
    "Age",
    min_value=10,
    max_value=75,
    value=30,
    step=1
)

# Prediction
if st.button("Predict Insurance Purchase"):
    input_data = pd.DataFrame([[age]], columns=['age'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("This person is likely to BUY life insurance.")
    else:
        st.error("This person is NOT likely to buy life insurance.")

    st.write(f"Probability of NOT buying insurance: {probability[0]:.2f}")
    st.write(f"Probability of BUYING insurance: {probability[1]:.2f}")
