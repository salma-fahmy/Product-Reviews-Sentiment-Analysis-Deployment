import os
import re
import pickle
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import torch

# ---------------------------- Page Config (MUST be first) ----------------------------
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

# ---------------------------- Preprocessing / Cleaning ----------------------------
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

    def predict_single(self, text: str):
        cleaned = self.clean_fn(text)
        enc = self.tokenizer(cleaned, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
        return {"pred_id": pred_id}

DEFAULT_ID2LABEL = {0:"negative", 1:"neutral", 2:"positive"}

# ---------------------------- Load Model ----------------------------
# Google Drive file ID for roberta_pipeline.pkl
GDRIVE_FILE_ID = "1Aggf2Hl1gEIE99W0T6I93lgDcTVRxI1N"
pipeline = None

@st.cache_resource
def load_model_from_gdrive_direct(file_id: str):
    """Load model directly from Google Drive into memory without saving locally."""
    try:
        import requests
        from io import BytesIO
        
        # Direct download URL for Google Drive
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Stream the file directly into memory
        response = requests.get(url, stream=True)
        
        # Handle Google's virus scan warning for large files
        if 'download_warning' in response.text or response.status_code != 200:
            # Try alternative method for large files
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Check for confirmation token
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'confirm': value, 'id': file_id}
                    response = session.get(url, params=params, stream=True)
                    break
        
        if response.status_code == 200:
            # Load pickle directly from memory
            file_buffer = BytesIO(response.content)
            model = pickle.load(file_buffer)
            return model, True
        else:
            return None, False
            
    except ImportError:
        return None, False
    except Exception as e:
        return None, False

# Load the model directly from Google Drive (no local file)
pipeline, MODEL_LOADED = load_model_from_gdrive_direct(GDRIVE_FILE_ID)

# ---------------------------- Styling ----------------------------
page_bg = """
<style>
/* Background for the entire app */
.stApp {
    background: 
        url('https://i.pinimg.com/736x/8a/4c/31/8a4c3184c5ae66e9f090f49db6bd445a.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #111827;
}


/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}

/* Text inside components */
.st-bq, .st-cy, .st-co, .stText, .stMarkdown {
    color: #111827 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.5) !important; /* semi-transparent */
    font-size: 18px !important;
    font-weight: bold !important;
    color: #111827 !important; /* dark text */
}

/* Sidebar radio button labels - make them black and bold */
[data-testid="stSidebar"] label {
    color: black !important;
    font-weight: bold !important;
}

/* Sidebar radio button text */
[data-testid="stSidebar"] .stRadio label p {
    color: black !important;
    font-weight: bold !important;
}

/* Sidebar title text */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: black !important;
    font-weight: bold !important;
}

/* Sidebar toggle button */
button[title="Collapse"] svg,
button[title="Expand"] svg {
    color: #111827 !important;  /* black icon */
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
    color: black;
    font-weight: bold;
    font-size: 18px;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    cursor: pointer;
    display: block;
    margin-left: auto;
    margin-right: auto;
    transition: 0.2s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #60A5FA, #93C5FD);
    transform: scale(1.03);
}

/* Inputs and DataFrames */
.stSelectbox, .stNumberInput, .stTextInput, .stDataFrame, .stFileUploader {
    background-color: rgba(255, 255, 255, 0.7);
    border-radius: 10px;
    padding: 10px;
    color: #111827 !important;
}

/* Forms */
.stForm {
    border: 1px solid rgba(0, 0, 0, 0.15);
    border-radius: 10px;
    padding: 10px;
}

/* Horizontal line */
hr {
    border-top: 1px solid rgba(0, 0, 0, 0.3) !important;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------- Header ----------------------------
st.markdown("<h1 style='text-align:center; color:black; '>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:black;'>Analyze single text or batch CSV files for sentiment.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------- Sidebar ----------------------------
st.sidebar.title("Configuration")
input_mode = st.sidebar.radio("Select input mode:", ['Single Text', 'Batch CSV'])

# ---------------------------- Single Text Mode ----------------------------
if input_mode == 'Single Text' and pipeline:
    st.subheader("Single Text Prediction")
    text_input = st.text_area("Enter text to classify", height=120)
    if st.button("🔍 Predict"):
        if not text_input.strip():
            st.warning("Please enter text.")
        else:
            try:
                res = pipeline.predict_single(text_input)
                pred_label = getattr(pipeline, "id2label", DEFAULT_ID2LABEL).get(res["pred_id"], str(res["pred_id"]))
                st.write("**Prediction ID:**", res["pred_id"])
                st.write("**Prediction Label:**", pred_label)
            except Exception as e:
                st.error(f"Inference failed: {e}")

# ---------------------------- Batch CSV Mode ----------------------------
elif input_mode == 'Batch CSV' and pipeline:
    st.subheader("Batch Prediction (CSV)")
    csv_file = st.file_uploader("Upload CSV file", type=["csv"])

    text_column_name = "Text"
    batch_size = 64

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head(5))
            if text_column_name not in df.columns:
                st.error(f"Column '{text_column_name}' not found.")
            else:
                if st.button("🚀 Run Batch Prediction"):
                    texts = df[text_column_name].astype(str).tolist()
                    preds = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        for t in batch:
                            if t.strip() == "" or t.lower() == "nan":
                                preds.append("empty")
                            else:
                                p = pipeline.predict_single(t)
                                preds.append(DEFAULT_ID2LABEL.get(p["pred_id"], str(p["pred_id"])))

                    df["pred_label"] = preds

                    st.success("Batch prediction completed.")
                    st.dataframe(df.head(10))

                    # ---------------- Prediction Counts صفين وعمودين ----------------
                    counts = df["pred_label"].value_counts()
                    st.info("### Prediction Counts")
                    all_labels = list(DEFAULT_ID2LABEL.values()) + ["empty"]

                    col1, col2 = st.columns(2)
                    for i, label in enumerate(all_labels):
                        label_text = f"{label.capitalize()}: {counts.get(label, 0)}"
                        if i % 2 == 0:
                            col1.write(label_text)
                        else:
                            col2.write(label_text)

                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions_with_labels.csv")
        except Exception as e:
            st.error(f"Failed to read/process CSV: {e}")

# ---------------------------- Footer ----------------------------
st.markdown("---")
st.caption("💡 Use this tool to understand the overall sentiment behind product reviews.")

