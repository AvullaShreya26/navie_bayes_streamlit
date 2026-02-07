import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Naive Bayes Classification App",
    page_icon="🌼",
    layout="centered"
)


# ------------------ Load CSS ------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ------------------ Title ------------------
st.title("🌼 Naive Bayes Classification")
st.write(
    "This application demonstrates **Naive Bayes classification** "
    "using the **Iris dataset**."
)


# ------------------ Load Dataset ------------------
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["Target"] = y


# ------------------ Dataset Preview ------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Select test size (%)",
    min_value=10,
    max_value=40,
    value=20
)


# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=42
)


# ------------------ Naive Bayes Model ------------------
model = GaussianNB()
model.fit(X_train, y_train)


# ------------------ Prediction & Accuracy ------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ------------------ Model Performance ------------------
st.subheader("📈 Model Performance")
st.success(f"Accuracy: {accuracy:.2f}")


# ------------------ User Input ------------------
st.subheader("🧪 Predict Flower Type")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)


# ------------------ Predict Button ------------------
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    flower_name = iris.target_names[prediction[0]]

    st.success(f"🌸 Predicted Flower: **{flower_name.capitalize()}**")


# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<center>Developed using Streamlit & Scikit-Learn</center>",
    unsafe_allow_html=True
)
