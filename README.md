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
python scripts/compare_models.py     # trains both models, picks thresholds
python scripts/benchmark_latency.py  # measures batch-inference latency
streamlit run app/dashboard.py       # opens the dashboard
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

## Baseline model

`python scripts/train_baseline.py` trains a Logistic Regression on the 28 PCA features plus log-transformed amount.

- **Time-based split:** the model trains on the earliest 80% of transactions and is tested on the latest 20%, mirroring production, where a model always scores future transactions.
- **Duplicates removed** (1,081 rows) so identical rows can't appear in both train and test.
- **Main metric is PR-AUC,** not accuracy, because of the 0.17% fraud rate.

Test set: 56,746 transactions, 74 fraud. Threshold 0.5.

| Model | PR-AUC | Recall | Precision | False-positive rate | Accuracy |
|---|---|---|---|---|---|
| Always "not fraud" | 0.001 | 0.0% | — | 0.00% | 99.87% |
| Logistic Regression | 0.751 | 55.4% | 91.1% | 0.01% | 99.93% |
| Logistic Regression, class-weighted | 0.750 | 90.5% | 4.0% | 2.83% | 97.16% |

Class weighting doesn't change how well the model *ranks* transactions (same PR-AUC). It shifts the trade-off: it catches 67 of 74 frauds instead of 41, but raises 1,605 false alarms. Choosing the right decision threshold is the next step.

## Model comparison and threshold selection

`python scripts/compare_models.py` compares class-weighted Logistic Regression with a Random Forest.

**Choosing the threshold.** In practice, a fraud team can only review so many alerts, so the threshold is set by an **alert budget**: at most 0.1% of legitimate transactions may be flagged. To avoid tuning on the test set, the training period is split again by time:

| Period | Transactions | Fraud |
|---|---|---|
| Train | 181,584 | 347 |
| Validation (threshold chosen here) | 45,396 | 52 |
| Test (reported below) | 56,746 | 74 |

Test results, with the threshold picked on validation:

| Model | PR-AUC | Frauds caught | Recall | Precision | False alarms | False-positive rate | Fraud $ caught |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.732 | 58 / 74 | 78.4% | 58.0% | 42 | 0.074% | 69.0% |
| Random Forest | **0.804** | **60 / 74** | **81.1%** | 48.0% | 65 | 0.115% | 69.2% |

- Random Forest ranks fraud better (PR-AUC 0.804 vs 0.732) and catches 2 more frauds.
- On the test period its false-positive rate (0.115%) came in slightly **over** the 0.1% budget it met on validation. The score distribution shifted between periods, which is why thresholds need ongoing monitoring.
- Both models catch about 69% of fraud dollars. The frauds they miss are disproportionately high-value, a natural target for the next iteration.

`reports/threshold_curve.csv` holds the full recall / false-alarm trade-off for each model, for the dashboard.

## Dashboard

`streamlit run app/dashboard.py` opens a three-tab dashboard:

- **Portfolio risk:** headline fraud KPIs and fraud rate by amount band and hour.
- **Model & threshold:** a threshold slider that updates frauds caught, false alarms, precision and fraud dollars caught live, with precision-recall curves for both models and the false-alarm trade-off.
- **Monitoring:** data-quality check status, alert-rate change from validation to test, feature drift measured with the Population Stability Index (PSI), and inference latency.

Findings from monitoring:

- Several features drift significantly between the training and test periods (V1 has PSI 1.35, well above the 0.25 "significant" line). The test period covers different hours of the day than training, so part of this likely reflects the daily cycle of card activity.
- The Random Forest's alert rate rose from 0.194% on validation to 0.220% on test at the same threshold.
- Batch inference on 1,000 transactions takes a median of 25.3 ms (Random Forest) and 0.6 ms (Logistic Regression) on an Apple Silicon laptop.

## Status

Core pipeline, models, and dashboard complete.
