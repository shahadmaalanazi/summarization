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

    inputs = tokenizer(
        "summarize: " + text,
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

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



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
# Streamlit UI
# =========================

st.title("📚 وش العلوم؟")
st.subheader("AI Text Summarization System")

st.markdown(
    """
This application summarizes long texts using **English T5**, **Arabic mT5**, and **TextRank**.
It supports both **English and Arabic** input text, with automatic language detection.
"""
)

st.info(
    """
- **T5 / mT5**: Abstractive summarization model  
- **TextRank**: Extractive summarization baseline for English text  
- **Evaluation**: ROUGE, Precision, Recall, and F1-score  
"""
)

# =========================
# Input Section
# =========================

with st.container():
    st.header("📝 Input Text")

    input_text = st.text_area(
        "Enter the article or paragraph you want to summarize:",
        height=250,
        placeholder="Paste your English or Arabic text here..."
    )

    reference_summary = st.text_area(
        "Reference Summary (Optional - for evaluation):",
        height=100,
        placeholder="Paste a human-written reference summary here if available..."
    )

    generate_button = st.button(
        "🚀 Generate Summary",
        use_container_width=True
    )


# =========================
# Run System
# =========================

if generate_button:
    if input_text.strip() == "":
        st.warning("Please enter text before generating a summary.")

    else:
        with st.spinner("Generating summaries... Please wait."):
            results = run_summarization_pipeline(input_text, reference_summary)

        st.success("Summary generated successfully!")

        # =========================
        # Basic Info
        # =========================

        st.divider()
        st.header("🔎 Detected Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Detected Language", results["language"])

        with col2:
            st.metric("Response Time", f"{results['response_time']:.2f} seconds")


        # =========================
        # Summaries
        # =========================

        st.divider()
        st.header("📊 Generated Summaries")

        tab1, tab2 = st.tabs(["🧠 T5 / mT5 Summary", "📌 TextRank Summary"])

        with tab1:
            st.subheader("Abstractive Summary")
            st.write(results["t5_summary"])

        with tab2:
            st.subheader("Extractive Summary")
            st.write(results["textrank_summary"])


        # =========================
        # Evaluation
        # =========================

        st.divider()
        st.header("📈 Evaluation Results")

        if reference_summary.strip() == "":
            st.warning(
                "No reference summary provided. Evaluation metrics cannot be calculated."
            )

        else:
            eval_tab1, eval_tab2 = st.tabs(
                ["🧠 T5 / mT5 Evaluation", "📌 TextRank Evaluation"]
            )

            with eval_tab1:
                st.subheader("T5 / mT5 ROUGE Scores")

                rouge_t5 = results["rouge_t5"]
                overlap_t5 = results["overlap_t5"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("ROUGE-1", format_metric(rouge_t5["ROUGE-1"]))
                with col2:
                    st.metric("ROUGE-2", format_metric(rouge_t5["ROUGE-2"]))
                with col3:
                    st.metric("ROUGE-L", format_metric(rouge_t5["ROUGE-L"]))

                st.subheader("T5 / mT5 Token Overlap Metrics")

                col4, col5, col6 = st.columns(3)

                with col4:
                    st.metric("Precision", format_metric(overlap_t5["Precision"]))
                with col5:
                    st.metric("Recall", format_metric(overlap_t5["Recall"]))
                with col6:
                    st.metric("F1-score", format_metric(overlap_t5["F1-score"]))


            with eval_tab2:
                st.subheader("TextRank ROUGE Scores")

                rouge_tr = results["rouge_textrank"]
                overlap_tr = results["overlap_textrank"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("ROUGE-1", format_metric(rouge_tr["ROUGE-1"]))
                with col2:
                    st.metric("ROUGE-2", format_metric(rouge_tr["ROUGE-2"]))
                with col3:
                    st.metric("ROUGE-L", format_metric(rouge_tr["ROUGE-L"]))

                st.subheader("TextRank Token Overlap Metrics")

                col4, col5, col6 = st.columns(3)

                with col4:
                    st.metric("Precision", format_metric(overlap_tr["Precision"]))
                with col5:
                    st.metric("Recall", format_metric(overlap_tr["Recall"]))
                with col6:
                    st.metric("F1-score", format_metric(overlap_tr["F1-score"]))


        # =========================
        # Model Explanation
        # =========================

        st.divider()
        st.header("🧩 Model Explanation")

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

st.caption(
    "وش العلوم؟ | AI Text Summarization System | T5 + mT5 + TextRank"
)
