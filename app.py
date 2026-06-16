"""
app.py — minimal Streamlit interface for the fake-job-posting baseline model.

Paste a job posting -> the saved pipeline (TF-IDF -> TruncatedSVD -> LogisticRegression)
predicts whether it looks real or fake. This is iteration 0 (deliberately simple).

Run:  streamlit run app.py
"""
import joblib
import streamlit as st

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")


@st.cache_resource  # load the saved model once and reuse it across reruns
def load_model():
    return joblib.load("fake_job_model.joblib")


model = load_model()

st.title("🕵️ Fake Job Posting Detector")
st.write(
    "Paste a job posting below and check whether it looks **real** or **fake**. "
    "_(Baseline model — iteration 0. It is intentionally simple and still misses many fakes.)_"
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
        # The pipeline expects an iterable of texts; we pass one and take the first result.
        prediction = model.predict([text])[0]
        fake_prob = model.predict_proba([text])[0][1]  # index 1 = probability of "fake"

        if prediction == 1:
            st.error(f"⚠️ Likely **FAKE** — {fake_prob:.0%} fraud probability")
        else:
            st.success(f"✅ Looks **real** — {fake_prob:.0%} fraud probability")

        st.caption("Model: TF-IDF → TruncatedSVD → Logistic Regression (baseline). Demo only.")
