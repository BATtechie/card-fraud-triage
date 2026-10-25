from pathlib import Path


SEED = 42
RAW_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"
ARTIFACTS = "artifacts"
CLEAN_PATH = "data/processed/creditcard_clean.csv"

Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(ARTIFACTS).mkdir(parents=True, exist_ok=True)
