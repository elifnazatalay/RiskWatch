"""Project paths and dataset constants shared across modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "riskwatch.db"

TABLE_NAME = "transactions"

# The Kaggle dataset: Time, 28 anonymized PCA features (V1..V28), Amount, Class.
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS = ["Time", *PCA_FEATURES, "Amount", "Class"]
LABEL = "Class"
