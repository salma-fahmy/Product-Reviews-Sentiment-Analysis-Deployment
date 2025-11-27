import streamlit as st
import pandas as pd
import torch
import pickle
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("Sentiment Analysis App")
st.markdown("---")

# ==============================
# NEW: Dropbox Direct Download
# ==============================
DROPBOX_URL = "https://www.dropbox.com/scl/fi/4r5mrc3tcrthzvstjpwjn/roberta_pipeline.pkl?rlkey=i5vli1htkljftqqcou8myu8y5&st=xyk2aahu&dl=1"


@st.cache_resource
def load_model_from_dropbox(url: str):
    try:
        st.write("Downloading model from Dropbox...")

        response = requests.get(url)
        response.raise_for_status()

        file_buffer = BytesIO(response.content)
        model = pickle.load(file_buffer)

        st.success("Model Loaded Successfully!")
        return model, True

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, False


pipeline, MODEL_LOADED = load_model_from_dropbox(DROPBOX_URL)

st.write("MODEL_LOADED:", MODEL_LOADED)

# =====================================================================
# UI SECTION
# =====================================================================
st.sidebar.title("Input Options")
input_mode = st.sidebar.radio("Choose input type:", ["Single Text", "Batch CSV"])

# =====================================================================
# SINGLE TEXT MODE
# =====================================================================
if input_mode == "Single Text" and MODEL_LOADED and pipeline is not None:
    text_input = st.text_area("Enter your text:", height=150)

    if st.button("Analyze"):
        with st.spinner("Predicting..."):
            try:
                result = pipeline(text_input)
                st.success(f"Prediction: **{result}**")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# =====================================================================
# BATCH CSV MODE
# =====================================================================
elif input_mode == "Batch CSV" and MODEL_LOADED and pipeline is not None:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Text" not in df.columns:
            st.error("CSV must contain a column named 'Text'")
        else:
            if st.button("Run Batch Prediction"):
                with st.spinner("Running predictions on CSV..."):
                    try:
                        df["Prediction"] = df["Text"].apply(lambda x: pipeline(x))
                        st.success("Batch Prediction Completed!")
                        st.dataframe(df)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

# =====================================================================
# FAILSAFE: Model didn't load
# =====================================================================
else:
    st.warning("⚠ Waiting for model to load... If nothing shows, reload the page.")
