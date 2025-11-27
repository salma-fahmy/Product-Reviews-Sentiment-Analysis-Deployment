import os
import re
import pickle
from typing import List, Dict, Any
from io import BytesIO
import requests

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

# ---------------------------- Load Serialized Model from Dropbox ----------------------------
DROPBOX_URL = "https://www.dropbox.com/scl/fi/4r5mrc3tcrthzvstjpwjn/roberta_pipeline.pkl?rlkey=i5vli1htkljftqqcou8myu8y5&dl=1"
pipeline = None
MODEL_LOADED = False

try:
    response = requests.get(DROPBOX_URL)
    response.raise_for_status()
    pipeline = pickle.load(BytesIO(response.content))
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Failed to load model from Dropbox: {e}")
    MODEL_LOADED = False

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

# باقي الكود زي ما هو عندك، بدون تغيير أي شيء في الـ Streamlit UI أو الموديل inference
