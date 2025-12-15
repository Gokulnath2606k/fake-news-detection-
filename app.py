import streamlit as st
import pickle
import os
import numpy as np

# ----------------------------------------------------------
# Load Model + Vectorizer
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------

st.set_page_config(page_title="Fake News Detection App", layout="wide")

st.title("📰 Fake News Detection System")
st.write("Enter any news article and the model will classify it as **Fake** or **Real** based on ML model.")

text_input = st.text_area("Enter News Content Here", height=250)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform text
        text_vec = vectorizer.transform([text_input])

        # Predict
        prediction = model.predict(text_vec)[0]

        # Output box
        if prediction.lower() == "fake":
            st.error("🚫 This news seems **FAKE**!")
        else:
            st.success("✅ This news seems **REAL**!")

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.write("---")
st.write("Built with ❤️ using Streamlit and Machine Learning.")
