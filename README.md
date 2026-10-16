# RiskWatch

Fraud detection and risk monitoring dashboard built with Python, SQL,
scikit-learn, and Streamlit.

## Planned features

- Reproducible data ingestion into SQLite
- Automated data-quality checks
- Logistic Regression and Random Forest model comparison
- Precision-recall and decision-threshold analysis
- Interactive risk-monitoring dashboard
- Distribution-drift checks and inference benchmarks

## Project structure

```text
RiskWatch/
├── data/               # Local raw and processed data (not committed)
├── notebooks/          # Exploratory analysis
├── scripts/            # Data download and pipeline commands
├── src/riskwatch/      # Reusable application code
└── tests/              # Automated tests
```

## Setup

1. Download `creditcard.csv` from the
   [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   into `data/raw/`.
2. Create the environment and load the data:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/load_data.py   # runs data-quality checks, writes data/processed/riskwatch.db
pytest
```

## Status

Initial project setup in progress.
