import streamlit as st
import joblib
import re


# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("svm_model.pkl")


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Page title
st.title("Consumer Complaint Classification")

st.write(
    "Enter a consumer complaint and the system will predict "
    "the financial product or service category."
)


# User input
complaint = st.text_area(
    "Enter your complaint:",
    height=200
)


# Prediction button
if st.button("Predict Category"):

    if complaint.strip() == "":
        st.warning("Please enter a complaint.")

    else:
        # Clean the complaint
        cleaned_complaint = clean_text(complaint)

        # Convert text into TF-IDF
        complaint_vector = vectorizer.transform([cleaned_complaint])

        # Predict category
        prediction = model.predict(complaint_vector)[0]

        st.success("Predicted Category:")

        st.write(prediction)


# Model information
st.divider()

st.subheader("Model Performance")

st.write("Accuracy: 86.54%")
st.write("Macro F1: 71.35%")
st.write("Weighted F1: 86.78%")