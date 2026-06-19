"""
app.py — Streamlit interface for the fake-job-posting detector.

Paste a job posting -> the saved model predicts real vs. fake.

The model was trained on cleaned + lemmatised text, so the app applies the SAME
preprocessing to the user's input before predicting (otherwise the input would not
match what the model learned).

Run:  streamlit run app.py
"""
import re
from html import unescape

import joblib
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")


# ---- preprocessing: must match the notebook exactly ----
@st.cache_resource
def get_lemmatizer():
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return WordNetLemmatizer()


def clean_text(text):
    text = unescape(str(text))                      # &amp; -> &
    text = re.sub(r"<[^>]+>", " ", text)            # HTML tags -> space
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)            # remove e-mail addresses
    return text


def lemmatize_text(text):
    lem = get_lemmatizer()
    return " ".join(lem.lemmatize(w) for w in re.findall(r"[a-z]+", text.lower()))


@st.cache_resource
def load_model():
    return joblib.load("fake_job_model.joblib")


model = load_model()

# ---- UI ----
st.title("🕵️ Fake Job Posting Detector")
st.write(
    "Paste a job posting below and check whether it looks **real** or **fake**. "
    "The model (TF-IDF + Logistic Regression) was tuned to catch fraudulent postings."
)

text = st.text_area(
    "Job posting text",
    height=220,
    placeholder="Paste the job title and description here...",
)

if st.button("Check posting"):
    if not text.strip():
        st.warning("Please paste a job posting first.")
    else:
        # apply the same cleaning + lemmatising as during training
        processed = lemmatize_text(clean_text(text))
        prediction = model.predict([processed])[0]
        # fraud probability: predict_proba if available, else map the decision score
        if hasattr(model, "predict_proba"):
            fake_prob = model.predict_proba([processed])[0][1]
            prob_txt = f" — {fake_prob:.0%} fraud probability"
        else:
            prob_txt = ""

        if prediction == 1:
            st.error(f"⚠️ Likely **FAKE**{prob_txt}")
        else:
            st.success(f"✅ Looks **real**{prob_txt}")

        st.caption("Baseline NLP project — for demonstration, not a guarantee.")
