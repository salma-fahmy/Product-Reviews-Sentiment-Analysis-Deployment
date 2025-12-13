import os
import re
import pickle
from io import BytesIO
import requests
import streamlit as st
import pandas as pd
import torch
import google.generativeai as genai

# ---------------------------- Configure Gemini API ----------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_working_gemini_model():
    try:
        models = genai.list_models()
        valid_models = [m.name for m in models if hasattr(m, "supported_generation_methods") 
                        and "generateContent" in m.supported_generation_methods]
        if not valid_models: return None
        priority = ["flash", "lite", "pro", "gemini"]
        for p in priority:
            for m in valid_models:
                if p in m.lower(): return m
        return valid_models[0]
    except:
        return None

def summarize_review_with_gemini(cleaned_text: str) -> str:
    if not GEMINI_API_KEY: return "⚠️ Gemini API key not configured."
    try:
        model_name = get_working_gemini_model()
        if not model_name:
            return "❌ No Gemini model available that supports generateContent."
        model = genai.GenerativeModel(model_name)
        prompt = f"Summarize the following customer review briefly:\n\nReview: {cleaned_text}\n\nSummary:"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# ---------------------------- Text Preprocessing ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub("", text)
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ---------------------------- Inference Pipeline ----------------------------
class InferencePipeline:
    def __init__(self, model, tokenizer, clean_fn=clean_text, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.clean_fn = clean_fn
        self.max_length = max_length
        self.model.eval()

    def predict_single(self, text):
        cleaned = self.clean_fn(text)
        enc = self.tokenizer(cleaned, truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
        return {"pred_id": pred_id}

DEFAULT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# ---------------------------- Load Model from GitHub ----------------------------
GITHUB_URL = "https://github.com/salma-fahmy/Product-Reviews-Sentiment-Analysis-Deployment/raw/main/model_bundle.pkl"

@st.cache_resource
def load_pipeline():
    placeholder = st.empty()
    placeholder.info("Downloading model from GitHub…")

    try:
        response = requests.get(GITHUB_URL)
        response.raise_for_status()
        file_bytes = BytesIO(response.content)

        # ----------------------------
        # 1. استخدام map_location عند التحميل
        # ----------------------------
        # map_location هو المفتاح الرئيسي لنقل التخزين إلى CPU
        pipeline = torch.load(file_bytes, map_location=torch.device('cpu'), weights_only=False)

        # ----------------------------
        # 2. تعيين النموذج الداخلي (الذي تم تحميله بالفعل) إلى CPU مرة أخرى
        #    للتأكد من أن أي مكونات فرعية لا تزال تشير إلى CUDA
        # ----------------------------
        if hasattr(pipeline, "model"):
            # نقوم بالتحويل الصريح إلى CPU
            pipeline.model.to(torch.device('cpu'))

        # ----------------------------
        # 3. التحقق من أي state_dict داخلي قد يحتاج إلى تحويل
        #    هذا يضيف طبقة إضافية من الأمان للنماذج المعقدة
        # ----------------------------
        if hasattr(pipeline.model, 'state_dict'):
            state_dict = pipeline.model.state_dict()
            for key in state_dict:
                if state_dict[key].is_cuda:
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
            
            # إعادة تحميل الـ state_dict المحوّل إلى النموذج
            pipeline.model.load_state_dict(state_dict)


        placeholder.empty()
        st.success("✅ Model loaded successfully on CPU!")
        return pipeline

    except Exception as e:
        # ملاحظة: إذا كان الخطأ ما زال يظهر، فعلى الأغلب ستحتاج للعودة للحل رقم 1.
        placeholder.error(f"❌ Failed to load model: {e}")
        return None

pipeline = load_pipeline()

# ---------------------------- Streamlit Page Setup ----------------------------
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

# ---------------------------- Streamlit Page Setup ----------------------------
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

# ---------------------------- Streamlit Page Setup ----------------------------
st.set_page_config(page_title="Product Reviews Sentiment Analysis", layout="wide")

page_bg = """
<style>
.stApp {
    background: url('https://i.pinimg.com/736x/8a/4c/31/8a4c3184c5ae66e9f090f49db6bd445a.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
h1, h2, h3, h4 { color: #111827 !important; }
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.5) !important;
}

/* Text color black for all texts */
body, .stText, .stMarkdown, .css-1d391kg, .stButton, .stTextInput label, .stTextArea label {
    color: #000000 !important;
}

/* Make labels bold */
.stTextInput label, .stTextArea label, .stRadio label {
    font-weight: bold !important;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# ---------------------------- Header ----------------------------
st.markdown("<h1 style='text-align:center;'>Product Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#111;'>Analyze single text or batch CSV files.</p>", unsafe_allow_html=True)
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
            st.warning("Please enter text first.")
        else:
            try:
                # Get cleaned text (after clean_text, before tokenizing)
                cleaned_text = clean_text(text_input)
                
                # Run sentiment prediction
                result = pipeline.predict_single(text_input)
                pred_label = getattr(pipeline, "id2label", DEFAULT_ID2LABEL).get(
                    result["pred_id"], str(result["pred_id"])
                )

                # Display results
                st.success("✅ Prediction Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Prediction ID:**", result["pred_id"])
                    st.write("**Predicted Label:**", pred_label)
                
                with col2:
                    st.write("**Cleaned Text:**")
                    st.info(cleaned_text if cleaned_text else "(empty after cleaning)")
                
                # Generate AI summary using Gemini
                st.write("---")
                st.write("Summary")
                
                with st.spinner("Generating summary with Gemini..."):
                    summary = summarize_review_with_gemini(cleaned_text)
                    st.write(summary)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


# ---------------------------- Batch CSV Mode ----------------------------
elif input_mode == "Batch CSV" and pipeline:
    st.subheader("Batch Prediction (CSV File)")
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    text_column = "Text"
    batch_size = 64

    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("### Preview")
            st.dataframe(df.head())

            if text_column not in df.columns:
                st.error(f"Column '{text_column}' is missing.")
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
                                label = DEFAULT_ID2LABEL.get(pred["pred_id"], str(pred["pred_id"]))
                                predictions.append(label)

                    df["pred_label"] = predictions

                    st.success("Batch prediction completed successfully.")
                    st.dataframe(df.head(10))

                    # ---------- Prediction Summary ----------
                    counts = df["pred_label"].value_counts()
                    st.info("### Prediction Summary")

                    all_labels = list(DEFAULT_ID2LABEL.values()) + ["empty"]
                    col1, col2 = st.columns(2)

                    for i, label in enumerate(all_labels):
                        text = f"{label.capitalize()}: {counts.get(label, 0)}"
                        if i % 2 == 0:
                            col1.write(text)
                        else:
                            col2.write(text)

                    # Download results
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV", data=csv_data, file_name="sentiment_predictions.csv")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# ---------------------------- Footer ----------------------------
st.markdown("---")
st.caption("💡 This app predicts sentiment for product reviews using a fine-tuned RoBERTa model.")













