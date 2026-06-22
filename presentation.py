"""
presentation.py — online presentation for the Fake Job Posting Detector.

A clickable Streamlit deck:
  1. Hook  2. Problem & data  3. Approach  4. Findings  5. Live demo  6. About me

Run:  streamlit run presentation.py
"""
import re
from html import unescape

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Fake Job Posting Detector", layout="centered")

# ---------------------------------------------------------------- pastel palette
TEAL, TEAL_LT, TEAL_DK = "#6FC5B8", "#E7F5F2", "#2F8C80"
LAV,  LAV_LT,  LAV_DK  = "#B49FDD", "#EFEAF8", "#6A50A0"
APR,  APR_LT,  APR_DK  = "#F1B583", "#FCEBDA", "#C2772F"

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: #FFFFFF; }}
      html, body, .stApp, p, li, span, div {{
          color: #000000;
          font-family: "Avenir", "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      }}
      h1, h2, h3, h4, h5 {{ color:#000 !important; text-align:center; font-weight:700;
          font-family:"Avenir","Avenir Next",sans-serif; }}
      .lead {{ text-align:center; font-size:1.3rem; line-height:1.65; }}
      .sub  {{ text-align:center; font-size:1.0rem; color:#444; }}
      .bignum {{ text-align:center; font-size:3.4rem; font-weight:800; color:{APR_DK}; }}
      .center {{ text-align:center; }}
      .pill {{ display:inline-block; background:{TEAL_LT}; color:{TEAL_DK}; border:1px solid {TEAL};
          border-radius:999px; padding:0.2rem 0.9rem; font-size:0.8rem; font-weight:600;
          letter-spacing:0.04em; }}
      /* framed text box for the demo */
      .stTextArea textarea {{ border:2px solid {TEAL} !important; border-radius:10px !important;
          background:{TEAL_LT}33 !important; }}
      .stButton > button {{ background:{TEAL}; color:#fff; border:none; border-radius:8px;
          padding:0.5rem 1.4rem; font-weight:600; }}
      .stButton > button:hover {{ background:{TEAL_DK}; color:#fff; }}
      thead th {{ background:{TEAL_LT} !important; color:#000 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def bar(color=TEAL):
    st.markdown(
        f"<div style='width:64px;height:4px;background:{color};border-radius:2px;"
        f"margin:0.1rem auto 1.3rem auto;'></div>", unsafe_allow_html=True)


def card(html, border=TEAL, bg=TEAL_LT):
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};border-radius:14px;"
        f"padding:1.1rem 1.35rem;margin:0.5rem 0;font-size:1.1rem;line-height:1.55;'>{html}</div>",
        unsafe_allow_html=True)


def stat(num, label, accent, bg):
    st.markdown(
        f"<div style='background:{bg};border:1px solid {accent};border-radius:14px;padding:1rem;"
        f"text-align:center;'><div style='font-size:2.1rem;font-weight:800;color:{accent}'>{num}</div>"
        f"<div class='sub'>{label}</div></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- real pipeline results
MODEL_RESULTS = pd.DataFrame(
    [["LinearSVC", 0.883, 0.803, 0.989, 0.990],
     ["LogReg", 0.881, 0.815, 0.987, 0.989],
     ["RandomForest", 0.746, 0.618, 0.970, 0.980],
     ["MultinomialNB", 0.721, 0.566, 0.981, 0.979],
     ["GradientBoosting", 0.628, 0.474, 0.954, 0.973]],
    columns=["Model", "F1 (fake)", "Recall (fake)", "ROC-AUC", "Accuracy"])

IMPROVEMENT = pd.DataFrame(
    [["0 - baseline (TF-IDF + SVD + LogReg)", 0.337, 0.202, 0.912, 0.961],
     ["1 - class_weight = balanced", 0.343, 0.861, 0.927, 0.840],
     ["2 - cleaning + model search (final)", 0.883, 0.803, 0.989, 0.990]],
    columns=["Step", "F1 (fake)", "Recall (fake)", "ROC-AUC", "Accuracy"])


# ---------------------------------------------------------------- demo model + preprocessing
@st.cache_resource
def get_lemmatizer():
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return WordNetLemmatizer()


def clean_text(text):
    text = unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    return text


def lemmatize_text(text):
    lem = get_lemmatizer()
    return " ".join(lem.lemmatize(w) for w in re.findall(r"[a-z]+", text.lower()))


@st.cache_resource
def load_model():
    return joblib.load("fake_job_model.joblib")


# ---------------------------------------------------------------- pages
def page_hook():
    st.markdown("<p class='center'><span class='pill'>IRONHACK · DATA PROJECT</span></p>", unsafe_allow_html=True)
    st.markdown("# Fake Job Posting Detector")
    bar()
    st.markdown("<p class='lead'>Catching scam job ads with Natural Language Processing and Machine Learning.</p>", unsafe_allow_html=True)
    st.markdown("<div class='bignum'>1 in 20</div>", unsafe_allow_html=True)
    st.markdown("<p class='sub'>job ads online is a scam — built to harvest personal data or money.</p>", unsafe_allow_html=True)
    st.markdown("<p class='lead'>Can a model flag them automatically, before they reach an applicant?</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub'>Daniel Vasic &middot; June 2026</p>", unsafe_allow_html=True)


def page_problem():
    st.markdown("## The Problem and the Data")
    bar(LAV)
    st.markdown("<p class='lead'>Fake job postings flood online platforms — stealing data, money and trust. Reviewing every ad by hand does not scale.</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: stat("17,880", "job postings", TEAL_DK, TEAL_LT)
    with c2: stat("4.8%", "are fraudulent", APR_DK, APR_LT)
    with c3: stat("4", "text fields used", LAV_DK, LAV_LT)
    card("<b>The core challenge is imbalance.</b> A model that calls everything \"real\" is already "
         "95% accurate — and useless. Finding the rare fakes is the whole task.", border=APR, bg=APR_LT)


def page_approach():
    st.markdown("## How It Was Built")
    bar(APR)
    steps = [
        ("1 · Clean & prepare text", "Remove HTML, URLs and e-mail addresses, then lemmatise (reduce words to their base form).", TEAL, TEAL_LT),
        ("2 · Word analysis", "Score each word: which ones separate real from fake, and which are just noise to remove.", LAV, LAV_LT),
        ("3 · Baseline → fix imbalance", "Start simple, then weight the rare fake class so it stops being ignored.", APR, APR_LT),
        ("4 · Systematic model search", "Five algorithms, each with a parameter grid, via cross-validated GridSearch — model choice, dimensionality reduction and tuning in one sweep.", TEAL, TEAL_LT),
    ]
    for title, body, br, bg in steps:
        card(f"<b>{title}</b><br><span class='sub' style='text-align:left'>{body}</span>", border=br, bg=bg)


def page_findings():
    st.markdown("## Findings")
    bar(LAV)
    card("Accuracy is misleading on imbalanced data, so the model was optimised for "
         "<b>F1 and recall of the fake class</b> — how many scams we actually catch.", border=LAV, bg=LAV_LT)

    st.markdown("##### Improvement over time")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    steps = ["baseline", "fix imbalance", "final"]
    ax.plot(steps, IMPROVEMENT["F1 (fake)"], color=TEAL, marker="o", linewidth=2.5, label="F1 (fake)")
    ax.plot(steps, IMPROVEMENT["Recall (fake)"], color=LAV, marker="s", linewidth=2.5,
            linestyle="--", label="Recall (fake)")
    ax.set_ylim(0, 1); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    st.table(IMPROVEMENT.set_index("Step"))
    st.markdown("##### Model comparison (full pipeline results)")
    st.table(MODEL_RESULTS.set_index("Model"))

    cc1, cc2, cc3 = st.columns(3)
    with cc1: stat("0.88", "F1 (fake)", TEAL_DK, TEAL_LT)
    with cc2: stat("0.80", "Recall (fake)", LAV_DK, LAV_LT)
    with cc3: stat("0.99", "ROC-AUC", APR_DK, APR_LT)

    card("<b>Best model: LinearSVC.</b> The simple linear model beat Random Forest and Gradient "
         "Boosting — dimensionality reduction blurred the rare fraud words.", border=TEAL, bg=TEAL_LT)
    st.markdown("<p class='sub'>Limitation: part of the signal comes from company names (source leakage), not pure fraud language.</p>", unsafe_allow_html=True)


def page_demo():
    st.markdown("## Live Demo")
    bar(APR)
    st.markdown("<p class='lead'>Paste any job posting and the model judges it in real time.</p>", unsafe_allow_html=True)
    model = load_model()
    text = st.text_area("Job posting text", height=200,
                        placeholder="Paste the job title and description here...")
    if st.button("Check posting"):
        if not text.strip():
            st.markdown("<p class='lead'>Please paste a posting first.</p>", unsafe_allow_html=True)
        else:
            processed = lemmatize_text(clean_text(text))
            pred = model.predict([processed])[0]
            if pred == 1:
                card("<h3 style='margin:0'>Prediction: FAKE</h3>", border=APR, bg=APR_LT)
            else:
                card("<h3 style='margin:0'>Prediction: REAL</h3>", border=TEAL, bg=TEAL_LT)


def page_about():
    st.markdown("## About Me")
    bar(TEAL)
    st.markdown("<p class='lead'><b>Daniel Vasic</b><br>Software developer (Fachinformatiker, IHK) "
                "specialising into Data &amp; Analytics. Strong in SQL and Python, building an "
                "end-to-end data portfolio.</p>", unsafe_allow_html=True)
    card("<b>Biggest challenge:</b> the class imbalance — learning why accuracy is the wrong metric.<br>"
         "<b>What is next:</b> remove company-name leakage, add a \"contains e-mail\" feature, deploy publicly.",
         border=LAV, bg=LAV_LT)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image("assets/linkedin_qr.png", caption="Connect on LinkedIn")
    st.markdown("<p class='center'><a href='https://www.linkedin.com/in/daniel-v-5a3312180'>"
                "linkedin.com/in/daniel-v-5a3312180</a></p>", unsafe_allow_html=True)


PAGES = {
    "1 - Hook": page_hook,
    "2 - Problem & data": page_problem,
    "3 - Approach": page_approach,
    "4 - Findings": page_findings,
    "5 - Live demo": page_demo,
    "6 - About me": page_about,
}

st.sidebar.title("Slides")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
