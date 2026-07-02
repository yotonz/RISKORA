# RiskOra — AI-Powered Credit Risk Intelligence Platform

RiskOra is a Streamlit web application that helps financial institutions assess loan default risk instantly. It combines a machine learning model with a rule-based override engine to produce explainable HIGH RISK / LOW RISK verdicts, backed by an interactive dashboard, EMI calculator, and AI assistant.

## Features

- **Instant Risk Assessment** — submit an applicant's age, income, loan amount, credit score, and EMI to get a real-time risk verdict.
- **Random Forest ML Model** — a 100-tree classifier trained on applicant data, combined with a deterministic rule engine (credit score, DTI, loan-to-income thresholds) for explainable overrides.
- **Financial Scoring** — a 0–100 financial health score derived from credit score, DTI ratio, and loan-to-income ratio.
- **EMI Calculator** — computes monthly installments and full amortization schedules for any loan amount, rate, and tenure.
- **Analytics Dashboard** — risk distribution charts, income vs. loan scatter plots, credit score histograms, DTI trends, and CSV export of applications.
- **AI Assistant** — a context-aware chat widget that answers questions about an applicant's risk factors, credit score bands, DTI, and loan eligibility.
- **Authentication & Roles** — user/admin accounts with PBKDF2-SHA256 password hashing (100,000 iterations, per-user salt); admins can view all applications, regular users see only their own.

## Tech Stack

| Layer | Technology |
|---|---|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| Machine learning | scikit-learn (Random Forest Classifier) |
| Data handling | pandas, numpy |
| Database | SQLite |
| Visualization | Matplotlib, Altair |
| Auth | PBKDF2-HMAC-SHA256 |

## Project Structure

```
RISKORA/
├── app.py              # Main Streamlit application (pages, UI, routing)
├── model.py            # Random Forest training & inference
├── db.py               # SQLite schema, queries, user/application storage
├── auth.py             # Password hashing utilities
├── chatbot.py          # Keyword-based AI assistant logic
├── utils/
│   ├── helpers.py       # Formatting & display helpers
│   ├── scoring.py        # DTI and financial score calculations
│   ├── rules.py          # Rule-based risk override engine
│   └── validation.py     # Input validation
├── data/
│   └── dataset.csv      # Training dataset for the ML model
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/yotonz/RISKORA.git
cd RISKORA
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`. A SQLite database (`database.db`) is created automatically on first run.

## How It Works

1. A user submits a loan application with their financial details.
2. Inputs are validated, and DTI (debt-to-income ratio) and a financial score are calculated.
3. The rule engine checks for hard-fail conditions (e.g., credit score below 550, DTI above 60%, loan more than 6× annual income).
4. The Random Forest model independently predicts risk based on the applicant's profile.
5. The combined verdict (HIGH RISK / LOW RISK) is shown with the contributing factors, and the application is stored for dashboard analytics.

## Disclaimer

RiskOra is a decision-support tool intended to assist risk analysis, not a substitute for a licensed lending authority's final decision.
