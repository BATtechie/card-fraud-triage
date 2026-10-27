## Setup

Requires Python 3.9 or higher.

### 1. Clone the repo
```bash
git clone https://github.com/BATtechie/card-fraud-triage.git
cd card-fraud-triage
```

### 2. Create and activate a virtual environment

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your terminal prompt should now start with `(.venv)`.

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register the Jupyter kernel (once)
```bash
python -m ipykernel install --user --name card-fraud --display-name "card-fraud (.venv)"
```
In VS Code or Antigravity, select this interpreter or kernel when opening a notebook.

### 5. Add the dataset
The dataset is not stored in this repo (it is about 150 MB). Download `creditcard.csv` from Kaggle ("Credit Card Fraud Detection" by ULB) and place it in `data/raw/`:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

Verify it:
```bash
python -c "import pandas as pd; print(pd.read_csv('data/raw/creditcard.csv').shape)"
```
Expected output: `(284807, 31)`

### 6. Run
1. Run data extraction and overview:
   Open `notebooks/extraction.ipynb` and run all cells.
2. Run data cleaning and audit reporting:
   Open `notebooks/cleaning.ipynb` and run all cells.
3. Run exploratory data analysis and feature profiling:
   Open `notebooks/eda.ipynb` and run all cells.
4. Run the AML GMM baseline:
   Open `notebooks/gmm_baseline.ipynb` and run all cells. The notebook audits the preserved `artifacts/gmm_model.joblib` and `artifacts/gmm_report.json`; it does not silently overwrite or retrain the selected artifact.

The GMM workflow uses a chronological 60/20/20 split. The preserved final
artifact was fitted on 169,893 legitimate training transactions only: training
frauds were excluded before fitting both the scaler and GMM. Validation-only
scores verify that the saved threshold is the 1% legitimate-FPR quantile. A
separate comparison GMM fitted on all training rows is explicitly labeled as
experimental and is not used for final model selection. The test set is not
read until the saved configuration and threshold have been selected and
audited. If BIC continues improving at a grid boundary,
the chosen model is only acceptable as the best validation candidate under
the stated compute budget—not as a globally optimal component count. The test
set is held out for final evaluation and does not select features or models.

Thresholds at 0.5%, 1%, and 2% target legitimate false-positive rates are
compared using legitimate validation scores. The notebook also reports
precision and recall at the top 100, 500, and 1,000 alerts. Since the project
does not specify the review team's capacity or tolerated missed fraud, the
configured 1% FPR is only a provisional reference; adjust
`TARGET_FALSE_POSITIVE_RATE` and review validation capacity results before
choosing an operating policy. A calibrated validation FPR does not guarantee
the same rate in future transactions.

Model candidates and thresholds are configurable in `src/config.py`. The
notebook reports stability across several fixed seeds, dataset hashes, code
revision, package versions, test metrics, and monitoring summaries.

The inference helper is `src.scoring.score_transactions`. It expects a pandas
DataFrame with the exact feature columns and order saved in the bundle. An
optional `Class` column is ignored; other unexpected columns are rejected.
Example after running the notebook:

```python
import joblib
import pandas as pd

from src.config import ARTIFACTS
from src.scoring import score_transactions

bundle = joblib.load(ARTIFACTS / "gmm_model.joblib")
new_transactions = pd.DataFrame(
    [[0.0] * len(bundle["feature_names"])],
    columns=bundle["feature_names"],
)
scores = score_transactions(new_transactions, bundle)
```

Only load joblib artifacts from trusted sources: joblib/pickle formats can
execute code when loaded. The scoring helper validates the fitted scaler,
GMM, feature order, and threshold metadata. It rejects missing, duplicate,
unexpected, non-numeric, and non-finite feature inputs; a supplied `Class`
column is ignored.

Run the scorer tests from the repository root with:

```bash
python -m pytest -q tests/
```

For monitoring, log scoring batch latency, anomaly-score quantiles, alert
counts/rate, and input-validation failures at the calling application
boundary. Compare feature and score distributions over time. Once labels
arrive, measure prevalence, false-positive rate, fraud precision, and fraud
recall by time period. Retraining and threshold recalibration should be
reviewed, evaluated, and versioned explicitly; do not automatically retrain
on incoming production transactions. Validate replacement artifacts before
rollout and retain a prior trusted artifact for rollback.

This GMM is an unsupervised anomaly detector, not a guaranteed fraud classifier
or calibrated fraud-probability model. New legitimate behavior may also
receive high anomaly scores. Results depend on the chronological holdout, the
cleaning choice to remove exact duplicate rows, the finite model-search budget,
and feature/covariance assumptions. A high recall can still mean substantial
false-alert review.

## Project structure
```
data/raw/        original dataset (not tracked by git)
data/processed/  generated splits (not tracked by git)
src/             reusable code (config, experiment, scoring, monitoring)
notebooks/       experiments and analysis
artifacts/       saved models, scores, metrics
reports/         figures and final report
tests/           sanity checks
```