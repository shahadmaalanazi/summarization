import os
# Dynamically locate the project directory. This ensures the app uses the D drive locally
# (solving the C drive full error) but remains fully portable for GitHub / Linux environments.
base_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["HF_HOME"] = os.path.join(base_dir, ".cache")

temp_dir = os.path.join(base_dir, ".tmp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir


import re
import time
import pandas as pd
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summa.summarizer import summarize
from rouge_score import rouge_scorer


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="وش العلوم | AI Text Summarization",
    page_icon="📚",
    layout="wide"
)


# =========================
# Model Names
# =========================

ENGLISH_MODEL_NAME = "t5-small"
ARABIC_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"


# =========================
# Load Models
# =========================

@st.cache_resource
def load_english_model():
    tokenizer = AutoTokenizer.from_pretrained(ENGLISH_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(ENGLISH_MODEL_NAME)
    return tokenizer, model


@st.cache_resource
def load_arabic_model():
    tokenizer = AutoTokenizer.from_pretrained(ARABIC_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(ARABIC_MODEL_NAME)
    return tokenizer, model
# =========================
# Text Helpers
# =========================

def clean_input(text):
    """
    Remove leading/trailing spaces and handle empty input safely.
    """
    if text is None:
        return ""
    return text.strip()


def is_arabic(text):
    """
    Detect if the input text contains Arabic characters.
    """
    arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
    return len(arabic_chars) > 0


def detect_language(text):
    """
    Returns Arabic or English based on the input characters.
    """
    if is_arabic(text):
        return "Arabic"
    return "English"

# =========================
# English T5 Summarization
# =========================

def english_t5_summary(text):

    tokenizer, model = load_english_model()

    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=30,
        num_beams=6,
        length_penalty=1.5,
        repetition_penalty=2.0,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# =========================
# Arabic / Multilingual mT5 Summarization
# =========================

def arabic_mt5_summary(text):
    tokenizer, model = load_arabic_model()

    # Preprocess text to handle whitespaces/newlines for mT5
    cleaned_text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', text.strip()))

    inputs = tokenizer(
        [cleaned_text],
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    summary_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)



# =========================
# Automatic T5 / mT5 Summary
# =========================

def t5_summary(text):
    """
    Automatically selects the English T5 model or Arabic multilingual mT5 model.
    """
    text = clean_input(text)

    if text == "":
        return "Please enter text to summarize."

    if is_arabic(text):
        return arabic_mt5_summary(text)

    return english_t5_summary(text)


# =========================
# TextRank Summary
# =========================

def textrank_summary(text):
    """
    TextRank is used as an extractive baseline.
    It works best with English multi-sentence text.
    """
    text = clean_input(text)

    if text == "":
        return "Please enter text to summarize."

    if is_arabic(text):
        return (
            "TextRank is mainly suitable for English extractive summarization in this project. "
            "Arabic summarization is handled using the multilingual mT5 model."
        )

    try:
        result = summarize(text, ratio=0.3)

        if result.strip() == "":
            return "Text too short for TextRank. Please enter a longer paragraph."

        return result

    except Exception:
        return "Text too short for TextRank or could not be summarized."
    
    # =========================
# ROUGE Evaluation
# =========================

def compute_rouge(reference, generated):
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """
    reference = clean_input(reference)
    generated = clean_input(generated)

    if reference == "":
        return {
            "ROUGE-1": None,
            "ROUGE-2": None,
            "ROUGE-L": None,
            "Message": "No reference summary provided."
        }

    if generated == "" or generated.startswith("Please enter"):
        return {
            "ROUGE-1": None,
            "ROUGE-2": None,
            "ROUGE-L": None,
            "Message": "No generated summary available."
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    return {
        "ROUGE-1": scores["rouge1"].fmeasure,
        "ROUGE-2": scores["rouge2"].fmeasure,
        "ROUGE-L": scores["rougeL"].fmeasure,
        "Message": "Evaluation completed."
    }


# =========================
# Precision / Recall / F1
# =========================

def compute_precision_recall_f1(reference, generated):
    """
    Simple token-overlap Precision, Recall, and F1-score.
    """
    reference = clean_input(reference).lower()
    generated = clean_input(generated).lower()

    if reference == "":
        return {
            "Precision": None,
            "Recall": None,
            "F1-score": None,
            "Message": "No reference summary provided."
        }

    ref_tokens = reference.split()
    gen_tokens = generated.split()

    if len(gen_tokens) == 0:
        return {
            "Precision": None,
            "Recall": None,
            "F1-score": None,
            "Message": "Generated summary is empty."
        }

    ref_set = set(ref_tokens)
    gen_set = set(gen_tokens)

    true_positive = len(ref_set.intersection(gen_set))

    precision = true_positive / len(gen_set)
    recall = true_positive / len(ref_set) if len(ref_set) > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Message": "Evaluation completed."
    }


# =========================
# Format Metrics
# =========================

def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"
# =========================
# Run Complete Summarization Pipeline
# =========================

def run_summarization_pipeline(input_text, reference_summary):
    """
    Runs:
    1. Language detection
    2. T5 / mT5 summarization
    3. TextRank summarization
    4. Evaluation
    """
    input_text = clean_input(input_text)
    reference_summary = clean_input(reference_summary)

    if input_text == "":
        return {
            "language": "N/A",
            "t5_summary": "Please enter text to summarize.",
            "textrank_summary": "Please enter text to summarize.",
            "rouge_t5": None,
            "rouge_textrank": None,
            "overlap_t5": None,
            "overlap_textrank": None,
            "response_time": 0
        }

    start_time = time.time()

    language = detect_language(input_text)

    generated_t5 = t5_summary(input_text)
    generated_textrank = textrank_summary(input_text)

    rouge_t5 = compute_rouge(reference_summary, generated_t5)
    rouge_textrank = compute_rouge(reference_summary, generated_textrank)

    overlap_t5 = compute_precision_recall_f1(reference_summary, generated_t5)
    overlap_textrank = compute_precision_recall_f1(reference_summary, generated_textrank)

    response_time = time.time() - start_time

    return {
        "language": language,
        "t5_summary": generated_t5,
        "textrank_summary": generated_textrank,
        "rouge_t5": rouge_t5,
        "rouge_textrank": rouge_textrank,
        "overlap_t5": overlap_t5,
        "overlap_textrank": overlap_textrank,
        "response_time": response_time
    }

# =========================
# Streamlit UI Style Injection (Premium Theme)
# =========================

st.set_page_config(
    page_title="وش العلوم؟ | AI Text Summarization System",
    page_icon="📚",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Tajawal:wght@300;400;500;700;800&display=swap');

    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Tajawal', 'Inter', sans-serif !important;
        background-color: #090d16 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(30, 58, 138, 0.25) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(13, 148, 136, 0.15) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(99, 102, 241, 0.12) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        font-family: 'Tajawal', 'Inter', sans-serif !important;
    }
    
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }
    
    h2, h3 {
        font-weight: 700 !important;
        color: #f8fafc !important;
        border-bottom: none !important;
    }

    /* Main Container Card */
    .stMainBlockContainer {
        max-width: 1200px !important;
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
    }

    /* Info Alerts */
    div[data-testid="stAlert"] > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] strong,
    div[data-testid="stAlert"] li,
    div[data-testid="stAlert"] div {
        color: #cbd5e1 !important;
    }
    
    /* Text Inputs & Textareas */
    [data-testid="stTextArea"] textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        padding: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.2) !important;
    }
    [data-testid="stTextArea"] label p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Generate Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.025em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #38bdf8 0%, #4f46e5 100%) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Metric Cards Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
    }

    /* Tab Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        background-color: rgba(15, 23, 42, 0.5) !important;
        padding: 8px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px !important;
        background-color: transparent !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        padding: 0 24px !important;
        border: none !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Table Styling */
    [data-testid="stDataFrame"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(15, 23, 42, 0.4) !important;
    }
    
    /* Custom divider styling */
    hr {
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.12), transparent) !important;
        margin: 3rem 0 !important;
    }
    
    /* Centered Header Layout */
    .main-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
    }
    
    .summary-card {
        background-color: rgba(15, 23, 42, 0.45);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        line-height: 1.8;
        color: #f1f5f9;
        font-size: 1.15rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📚 وش العلوم؟</h1>
        <p style="font-size: 1.35rem; color: #94a3b8; font-weight: 500; margin-top: 0.5rem;">AI Text Summarization System • نظام التلخيص الذكي للنصوص</p>
    </div>
    """,
    unsafe_allow_html=True
)

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(
        "🧠 **التلخيص التوليدي (Abstractive)**\n\nيستخدم نماذج **English T5** و **Arabic mT5** المتطورة لإعادة صياغة النص وتوليد ملخص ذكي وجديد بالكامل يركز على الفكرة الأساسية."
    )
with col_info2:
    st.info(
        "📌 **التلخيص الاستخراجي (Extractive)**\n\nيعتمد على خوارزمية **TextRank** لاستخلاص الجمل الأكثر أهمية مباشرة من النص الأصلي (يدعم اللغة الإنجليزية)."
    )

st.divider()

# =========================
# Input Section
# =========================

with st.container():
    st.markdown("### 📝 النص المراد تلخيصه")

    input_text = st.text_area(
        "أدخل المقال أو النص الذي تريد تلخيصه (يدعم العربية والإنجليزية تلقائياً):",
        height=250,
        placeholder="Paste your text here / الصق النص هنا..."
    )

    reference_summary = st.text_area(
        "الملخص المرجعي (اختياري - لحساب مقاييس الدقة والتقييم):",
        height=100,
        placeholder="Paste reference summary / الصق الملخص المرجعي للمقارنة هنا..."
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    generate_button = st.button(
        "🚀 توليد الملخص الذكي",
        use_container_width=True
    )


# =========================
# Run System
# =========================

if generate_button:
    if input_text.strip() == "":
        st.warning("الرجاء إدخال نص أولاً لتوليد الملخص.")

    else:
        with st.spinner("جاري معالجة وتوليد الملخصات... الرجاء الانتظار."):
            results = run_summarization_pipeline(input_text, reference_summary)

        st.success("تم توليد الملخصات بنجاح!")

        # =========================
        # Basic Info
        # =========================

        st.divider()
        st.markdown("### 🔎 معلومات النص المكتشفة")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Detected Language | اللغة المكتشفة", results["language"])

        with col2:
            st.metric("Response Time | زمن الاستجابة", f"{results['response_time']:.2f} ثواني")
            
        with col3:
            st.metric("Evaluation Status | حالة التقييم", "نشطة (مقارنة)" if reference_summary.strip() else "غير نشطة")


        # =========================
        # Summaries
        # =========================

        st.divider()
        st.markdown("### 📊 الملخصات الناتجة")

        tab1, tab2 = st.tabs(["🧠 ملخص الذكاء الاصطناعي (T5 / mT5)", "📌 ملخص استخراج الجمل (TextRank)"])

        with tab1:
            st.markdown("#### Abstractive Summary")
            st.markdown(
                f'<div class="summary-card" style="border-left: 5px solid #38bdf8;">{results["t5_summary"]}</div>',
                unsafe_allow_html=True
            )

        with tab2:
            st.markdown("#### Extractive Summary")
            st.markdown(
                f'<div class="summary-card" style="border-left: 5px solid #6366f1;">{results["textrank_summary"]}</div>',
                unsafe_allow_html=True
            )


        # =========================
        # Evaluation
        # =========================

        st.divider()
        st.markdown("### 📈 نتائج ودقة التقييم")

        if reference_summary.strip() == "":
            st.info(
                "لم يتم إدخال ملخص مرجعي، يمكنك إدخاله في صندوق الإدخال بالأعلى للمقارنة التلقائية وحساب قيم ROUGE والدقة."
            )

        else:
            eval_tab1, eval_tab2 = st.tabs(
                ["🧠 تقييم نموذج T5 / mT5", "📌 تقييم خوارزمية TextRank"]
            )

            with eval_tab1:
                st.markdown("#### ROUGE Scores")

                rouge_t5 = results["rouge_t5"]
                overlap_t5 = results["overlap_t5"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("ROUGE-1", format_metric(rouge_t5["ROUGE-1"]))
                with col2:
                    st.metric("ROUGE-2", format_metric(rouge_t5["ROUGE-2"]))
                with col3:
                    st.metric("ROUGE-L", format_metric(rouge_t5["ROUGE-L"]))

                st.markdown("#### Token Overlap Metrics")

                col4, col5, col6 = st.columns(3)

                with col4:
                    st.metric("Precision (الدقة)", format_metric(overlap_t5["Precision"]))
                with col5:
                    st.metric("Recall (الاسترجاع)", format_metric(overlap_t5["Recall"]))
                with col6:
                    st.metric("F1-score (مقياس إف)", format_metric(overlap_t5["F1-score"]))


            with eval_tab2:
                st.markdown("#### ROUGE Scores")

                rouge_tr = results["rouge_textrank"]
                overlap_tr = results["overlap_textrank"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("ROUGE-1", format_metric(rouge_tr["ROUGE-1"]))
                with col2:
                    st.metric("ROUGE-2", format_metric(rouge_tr["ROUGE-2"]))
                with col3:
                    st.metric("ROUGE-L", format_metric(rouge_tr["ROUGE-L"]))

                st.markdown("#### Token Overlap Metrics")

                col4, col5, col6 = st.columns(3)

                with col4:
                    st.metric("Precision (الدقة)", format_metric(overlap_tr["Precision"]))
                with col5:
                    st.metric("Recall (الاسترجاع)", format_metric(overlap_tr["Recall"]))
                with col6:
                    st.metric("F1-score (مقياس إف)", format_metric(overlap_tr["F1-score"]))


        # =========================
        # Model Explanation
        # =========================

        st.divider()
        st.markdown("### 🧩 تفاصيل النماذج المستخدمة")

        explanation_df = pd.DataFrame({
            "Model": ["T5-small", "mT5 multilingual XLSum", "TextRank"],
            "Type": [
                "Abstractive",
                "Abstractive / Multilingual",
                "Extractive"
            ],
            "Used For": [
                "English summarization",
                "Arabic summarization",
                "English extractive baseline"
            ],
            "Description": [
                "Generates new summary sentences based on input meaning.",
                "Supports multilingual summarization including Arabic.",
                "Selects important sentences directly from the original text."
            ]
        })

        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True
        )


# =========================
# Footer
# =========================

st.divider()

st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding-bottom: 2rem;">
        وش العلوم؟ | AI Text Summarization System | T5 + mT5 + TextRank
    </div>
    """,
    unsafe_allow_html=True
)
