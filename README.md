# Fake Job Posting Detector

Detect fraudulent job postings from their text using NLP + machine learning.

## Business case
Job platforms are flooded with scam postings that harvest personal data or money from
applicants. Reviewing every ad by hand does not scale. This project builds a classifier
that flags likely-fraudulent postings automatically, so a platform can prioritise them
for review before they reach applicants.

## Dataset
- Source: Kaggle — *Real / Fake Job Posting Prediction* (`shivamb/real-or-fake-fake-jobposting-prediction`)
- ~17,880 postings, **only ~4.8% fraudulent** → heavily imbalanced
- Columns used: `title`, `company_profile`, `description`, `requirements` (text) and `fraudulent` (target)

## Metric
**F1 of the fake class** is the headline metric, reported with **Recall (fake)** and
**ROC-AUC**. Accuracy is *not* used as the main metric: with 4.8% fakes, always
predicting "real" already scores ~95% — that hides whether the model actually catches
fraud. The expensive error is letting a fake through, so recall of the fake class matters.

## Approach (the pipeline)
1. **Text preprocessing** — combine the text fields, clean HTML/URLs/e-mail addresses,
   lemmatise.
2. **Word analysis** — per-class frequency + a signed "lean" score to separate signal
   words from neutral noise words (the latter become custom stop words).
3. **Baseline → improvement** — start with a deliberately simple model, then handle the
   class imbalance (`class_weight="balanced"`).
4. **Systematic model search** — 5 models (Logistic Regression, LinearSVC, Multinomial
   Naive Bayes, Random Forest, Gradient Boosting), each with a parameter grid, compared
   via `GridSearchCV` + `DataFrame.apply()` (model comparison + dimensionality reduction
   with TruncatedSVD + hyperparameter tuning in one sweep).

## Results — improvement over time
| Step | F1 (fake) | Recall (fake) | ROC-AUC |
|------|-----------|---------------|---------|
| 0 · baseline (TF-IDF + SVD + LogReg) | 0.34 | 0.20 | 0.91 |
| 1 · class_weight = balanced | 0.34 | 0.86 | 0.93 |
| 2 · cleaning + word analysis + model search | **0.88** | 0.80 | **0.99** |

**Best model: LinearSVC** (TF-IDF, `class_weight="balanced"`) — F1 (fake) ≈ 0.88,
ROC-AUC ≈ 0.99. The linear model on clean TF-IDF beat the tree models; TruncatedSVD
actually hurt them because it blurs the rare, telling fraud words.

## Limitation
Some of the strongest signal comes from **company / brand names** (a single employer's
ads all fall in one class). That is partly *source leakage*, not pure fraud language —
on genuinely mixed, real-world data the score would be lower.

## How to run
```bash
pip install -r requirements.txt

# the full analysis
jupyter notebook fake_job_detector.ipynb

# the interactive demo
streamlit run app.py
```

## Tools & tech
Python · pandas · scikit-learn · NLTK · matplotlib · Streamlit · joblib

## Repo structure
```
fake-job-detector/
├── data/                      # dataset (fake_job_postings.csv)
├── assets/                    # LinkedIn QR for the presentation
├── fake_job_detector.ipynb    # full pipeline: cleaning -> EDA -> model search -> best model
├── app.py                     # Streamlit demo (paste a posting -> real/fake)
├── presentation.py            # Streamlit slide deck for the project presentation
├── fake_job_model.joblib      # the saved best model (LinearSVC)
├── requirements.txt           # dependencies
└── README.md
```

## What I'd add next
- Remove company-name leakage from the features
- Add a "contains e-mail address" flag as an extra signal
- Deploy the demo publicly

## About me
**Daniel Vasic** — software developer (Fachinformatiker, IHK) specialising into Data & Analytics.
- LinkedIn: https://www.linkedin.com/in/daniel-v-5a3312180

_June 2026_
