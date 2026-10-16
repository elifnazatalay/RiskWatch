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
python scripts/load_data.py     # runs data-quality checks, writes data/processed/riskwatch.db
python scripts/run_analysis.py  # runs every query in sql/
pytest
```

## SQL analysis

| Query | Question it answers |
|---|---|
| [`01_overview`](sql/01_overview.sql) | How much fraud is there, by count and by dollar value? |
| [`02_amount_bands`](sql/02_amount_bands.sql) | Which transaction sizes are riskiest (fraud rate and lift)? |
| [`03_hourly_risk`](sql/03_hourly_risk.sql) | Which hours have the highest fraud rate? |
| [`04_fraud_concentration`](sql/04_fraud_concentration.sql) | How concentrated are fraud losses among the largest frauds? |
| [`05_daily_drift`](sql/05_daily_drift.sql) | Did fraud patterns shift between day 1 and day 2? |

Key findings:

- 492 of 284,807 transactions (0.173%) are fraud, so accuracy is a misleading metric: a model that never predicts fraud is 99.83% accurate.
- $0 transactions have 8.6x the average fraud rate, and 45% of all frauds are under $10, which is consistent with small "card-testing" charges.
- Fraud is concentrated in low-volume hours: hour 2 has a 1.71% fraud rate (about 10x average) while carrying only 1.2% of volume.
- The largest 10% of frauds account for 61% of fraud dollars, so catching high-value frauds matters more than raw fraud count.
- The fraud rate fell 22% from day 1 to day 2, a reminder that fraud patterns shift and models need monitoring.

## Status

Initial project setup in progress.
