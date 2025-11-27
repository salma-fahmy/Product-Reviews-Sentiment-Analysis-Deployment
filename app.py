import os
import re
import pickle
import requests
from typing import List, Dict, Any

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
        "]", flags=re.UNICODE)
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
            logits = out.logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
        return {"pred_id": pred_id}

DEFAULT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# ---------------------------- Load Serialized Model from Dropbox ------------
MODEL_URL = "https://www.dropbox.com/scl/fi/4r5mrc3tcrthzvstjpwjn/roberta_pipeline.pkl?dl=1"
MODEL_FILENAME = "roberta_pipeline.pkl"
pipeline = None

def download_model(url, filename):
    if os.path.exists(filename):
        return True
    try:
        st.info("Downloading model from Dropbox, please wait...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return False

# Load model
MODEL_LOADED = False
if download_model(MODEL_URL, MODEL_FILENAME):
    try:
        with open(MODEL_FILENAME, "rb") as f:
            pipeline = pickle.load(f)
        MODEL_LOADED = True
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.error("Model not available.")

# ---------------------------- Streamlit Page Setup ----------------------------
st.set_page_config(
    page_title="Product Reviews Sentiment Analysis",
    layout="wide"
)

page_bg = """
<style>
.stApp {background: url('https://i.pinimg.com/736x/8a/4c/31/8a4c3184c5ae66e9f090f49db6bd445a.jpg'); background-size: cover; background-position: center; background-repeat: no-repeat; color: #111827;}
h1,h2,h3,h4,h5,h6{color:#111827 !important;}
.st-bq,.st-cy,.st-co,.stText,.stMarkdown{color:#111827 !important;}
[data-testid="stSidebar"]{background-color: rgba(255,255,255,0.5) !important; font-size:18px !important; font-weight:bold !important; color:#111827 !important;}
.stButton>button{background:linear-gradient(90deg,#3B82F6,#60A5FA); color:black; font-weight:bold; font-size:18px; border-radius:10px; padding:0.6em 1.2em; transition:0.2s; cursor:pointer;}
.stButton>button:hover{background:linear-gradient(90deg,#60A5FA,#93C5FD); transform:scale(1.03);}
.stSelectbox,.stNumberInput,.stTextInput,.stDataFrame,.stFileUploader{background-color: rgba(255,255,255,0.7); border-radius:10px; padding:10px; color:#111827 !important;}
.stForm{border:1px solid rgba(0,0,0,0.15); border-radius:10px; padding:10px;}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#111827;'>Analyze single text reviews or batch CSV files.</p>", unsafe_allow_html=True)

# Use st.divider() instead of st.markdown("---") to avoid SyntaxError
st.divider()

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
                pred_label = getattr(pipeline, "id2label", DEFAULT_ID2LABEL).get(result["pred_id"])
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
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        for t in batch:
                            if t.strip() == "" or t.lower() == "nan":
                                predictions.append("empty")
                            else:
                                pred = pipeline.predict_single(t)
                                predictions.append(DEFAULT_ID2LABEL.get(pred["pred_id"], str(pred["pred_id"])))
                    df["pred_label"] = predictions
                    st.success("Batch prediction completed successfully.")
                    st.dataframe(df.head(10))
                    counts = df["pred_label"].value_counts()
                    st.info("### Prediction Summary")
                    all_labels = list(DEFAULT_ID2LABEL.values()) + ["empty"]
                    col1, col2 = st.columns(2)
                    for i, label in enumerate(all_labels):
                        text = f"{label.capitalize()}: {counts.get(label, 0)}"
                        (col1 if i % 2 == 0 else col2).write(text)
                    st.download_button("Download Results CSV", df.to_csv(index=False).encode("utf-8"), "sentiment_predictions.csv")

# Footer
st.divider()
st.caption("💡 This dashboard helps you understand the sentiment behind customer product reviews.")
