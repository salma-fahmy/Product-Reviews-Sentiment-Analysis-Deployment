import os
import re
import pickle
import gdown
from typing import Dict, Any

import streamlit as st
import pandas as pd
import torch

# ============================ Text Preprocessing ============================
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
        "]", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================ Inference Pipeline ============================
class InferencePipeline:
    def __init__(self, model, tokenizer, clean_fn=clean_text, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.clean_fn = clean_fn
        self.max_length = max_length
        self.model.eval()

    def predict_single(self, text: str) -> Dict[str, Any]:
        cleaned = self.clean_fn(text)
        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
        return {"pred_id": pred_id}

DEFAULT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# ============================ Load Model via gdown ===========================
MODEL_URL = "https://drive.google.com/uc?id=1Aggf2Hl1gEIE99W0T6I93lgDcTVRxI1N"
MODEL_FILENAME = "roberta_pipeline.pkl"
pipeline = None

if not os.path.exists(MODEL_FILENAME):
    st.info("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

try:
    with open(MODEL_FILENAME, "rb") as f:
        pipeline = pickle.load(f)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ============================ Streamlit UI ============================
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

st.markdown("<h1 style='text-align:center;'>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze individual reviews or CSV files.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("Configuration")
input_mode = st.sidebar.radio("Select Input Mode:", ["Single Text", "Batch CSV"])

# Single text mode
if input_mode == "Single Text" and pipeline:
    st.subheader("Single Review Prediction")
    text_input = st.text_area("Enter a review:", height=120)
    if st.button("🔍 Predict"):
        if not text_input.strip():
            st.warning("Please enter text before predicting.")
        else:
            try:
                res = pipeline.predict_single(text_input)
                label = getattr(pipeline, "id2label", DEFAULT_ID2LABEL).get(res["pred_id"])
                st.write("**Prediction ID:**", res["pred_id"])
                st.write("**Predicted Label:**", label)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Batch CSV mode
elif input_mode == "Batch CSV" and pipeline:
    st.subheader("Batch Prediction (CSV Upload)")
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])
    text_column = "Text"
    batch_size = 64
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("### Preview")
            st.dataframe(df.head())
            if text_column not in df.columns:
                st.error(f"Column '{text_column}' not found in file.")
            else:
                if st.button("🚀 Run Batch Prediction"):
                    texts = df[text_column].astype(str).tolist()
                    preds = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        for t in batch:
                            if t.strip() == "" or t.lower() == "nan":
                                preds.append("empty")
                            else:
                                p = pipeline.predict_single(t)
                                preds.append(DEFAULT_ID2LABEL.get(p["pred_id"]))
                    df["pred_label"] = preds
                    st.success("Batch prediction completed successfully.")
                    st.dataframe(df.head(10))
                    counts = df["pred_label"].value_counts()
                    st.info("### Prediction Summary")
                    col1, col2 = st.columns(2)
                    labels = ["negative", "neutral", "positive", "empty"]
                    for i, lbl in enumerate(labels):
                        text = f"{lbl.capitalize()}: {counts.get(lbl, 0)}"
                        (col1 if i % 2 == 0 else col2).write(text)
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV", data=csv_bytes, file_name="sentiment_predictions.csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.markdown("---")
st.caption("💡 This tool helps you evaluate customer review sentiment.")
