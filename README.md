# 📚 وش العلوم؟ | AI Text Summarization System

An AI-powered text summarization system that supports both Arabic and English languages using Transformer-based models and Natural Language Processing (NLP).

## 🚀 Features

- Automatic language detection (Arabic / English)
- Abstractive summarization using T5 and mT5 models
- Extractive summarization using TextRank
- ROUGE evaluation metrics
- Precision, Recall, and F1-score calculation
- Interactive Streamlit web application
- Arabic and English text support

## 🛠 Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- T5-small
- mT5 XLSum
- TextRank
- ROUGE Score
- Pandas
- NLP

## 📊 Models

| Model | Type | Purpose |
|---------|---------|---------|
| T5-small | Abstractive | English Summarization |
| mT5 XLSum | Abstractive | Arabic Summarization |
| TextRank | Extractive | Baseline Summarization |

## 📈 Evaluation Metrics

The system evaluates generated summaries using:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- Precision
- Recall
- F1-Score

## 🌐 Live Demo

https://summarization-npub4ktlswxnnbebhwxyue.streamlit.app/

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

pip install -r requirements.txt

streamlit run app.py
