from pathlib import Path


SEED = 42
GMM_CANDIDATES = {
    "diag": (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256),
    "spherical": (1, 2, 4, 8, 16, 32, 64),
    "tied": (1, 2, 4, 8, 16, 32),
    "full": (1, 2, 4, 8),
}
GMM_MAX_ITER = 150
GMM_N_INIT = 1
GMM_REG_COVAR = 1e-6
TARGET_FALSE_POSITIVE_RATE = 0.01
TARGET_FALSE_POSITIVE_RATES = (0.005, 0.01, 0.02)
ALERT_CAPACITIES = (100, 500, 1000)
STABILITY_SEEDS = (42, 43, 44, 45, 46)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_clean.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
