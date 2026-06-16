# Fake Job Posting Detector

Detect fraudulent job postings with NLP + machine learning.

## Business case
_(TODO: who needs this and why — e.g. a job platform wants to auto-flag scam postings before they reach applicants.)_

## Dataset
- Source: Kaggle — "Real / Fake Job Posting Prediction" (`shivamb/real-or-fake-fake-jobposting-prediction`)
- ~17,880 postings, ~4.8% fraudulent (heavily imbalanced)
- Columns: title, location, company_profile, description, requirements, benefits, several flags, and `fraudulent` (target)

## Approach
_(TODO: cleaning → EDA → feature engineering → models → evaluation. Fill in as you go.)_

## Results
_(TODO: best model, key metrics — focus on recall/F1 for the fake class, not just accuracy.)_

## How to run
```bash
pip install -r requirements.txt
jupyter notebook fake_job_detector.ipynb
```

## Repo structure
```
fake-job-detector/
├── data/                     # dataset
├── fake_job_detector.ipynb   # full pipeline
├── requirements.txt
└── README.md
```

## Author
Daniel Vasic · June 2026
