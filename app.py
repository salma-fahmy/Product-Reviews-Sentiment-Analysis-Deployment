import os
import re
import pickle
from typing import List, Dict, Any
import requests
from io import BytesIO

import streamlit as st
import pandas as pd
import torch

# ---------------------------- Text Preprocessing ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------- Inference Pipeline ----------------------------
class InferencePipeline:
    def __init__(self, model, tokenizer, clean_fn=clean_text, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.clean_fn = clean_fn
        self.max_length = max_length
        self.model.eval()

    def predict_single(self, text: str) -> Dict[str, Any]:
        cleaned = self.clean_fn(text)
        enc = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(**enc)
            pred_id = int(torch.argmax(out.logits, dim=-1).item())
        return {"pred_id": pred_id}


DEFAULT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


# ---------------------------- Load Serialized Model from Dropbox ----------------------------
DROPBOX_URL = "https://www.dropbox.com/scl/fi/4r5mrc3tcrthzvstjpwjn/roberta_pipeline.pkl?rlkey=i5vli1htkljftqqcou8myu8y5&st=xyk2aahu&dl=1"

@st.cache_resource
def load_pipeline():
    try:
        st.info("Downloading model from Dropbox…")
        response = requests.get(DROPBOX_URL)
        response.raise_for_status()

        file_bytes = BytesIO(response.content)
        pipeline = pickle.load(file_bytes)

        st.success("Model Loaded Successfully!")
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

pipeline = load_pipeline()


# ---------------------------- Streamlit Page Setup ----------------------------
st.set_page_config(
    page_title="Product Reviews Sentiment Analysis",
    layout="wide"
)

page_bg = """
<style>
.stApp {
    background: url('https://i.pinimg.com/736x/8a/4c/31/8a4c3184c5ae66e9f090f49db6bd445a.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #111827;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#111827;'>Analyze single text reviews or batch CSV files.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------- Sidebar ----------------------------
st.sidebar.title("Configuration")
input_mode = st.sidebar.radio("Select input mode:", ["Single Text", "Batch CSV"])

# ---------------------------- Single Text Mode ----------------------------
if input_mode == "Single Text" and pipeline:
    st.subheader("Single Review Prediction")
    text_input = st.text_area("Enter a review:", height=120)

    if st.button("🔍 Predict"):
        if not text_input.strip():
            st.warning("Please enter text before predicting.")
        else:
            try:
                result = pipeline.predict_single(text_input)
                pred_label = getattr(pipeline, "id2label", DEFAULT_ID2LABEL).get(result["pred_id"], str(result["pred_id"]))
                st.write("**Prediction ID:**", result["pred_id"])
                st.write("**Predicted Label:**", pred_label)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------- Batch CSV Mode ----------------------------
elif input_mode == "Batch CSV" and pipeline:
    st.subheader("Batch Prediction (CSV Upload)")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    text_column = "Text"
    batch_size = 64

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("### Preview")
            st.dataframe(df.head())

            if text_column not in df.columns:
                st.error(f"Column '{text_column}' not found in uploaded file.")
            else:
                if st.button("🚀 Run Batch Prediction"):
                    texts = df[text_column].astype(str).tolist()
                    predictions = []

                    for text in texts:
                        if text.strip() == "" or text.lower() == "nan":
                            predictions.append("empty")
                        else:
                            pred = pipeline.predict_single(text)
                            label = DEFAULT_ID2LABEL.get(pred["pred_id"], str(pred["pred_id"]))
                            predictions.append(label)

                    df["pred_label"] = predictions
                    st.success("Batch prediction completed successfully.")
                    st.dataframe(df.head(10))

                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV", data=csv_data, file_name="sentiment_predictions.csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.caption("💡 This dashboard helps you understand the sentiment behind customer product reviews.")
