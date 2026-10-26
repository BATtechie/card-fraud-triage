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

## Project structure
```
data/raw/        original dataset (not tracked by git)
data/processed/  generated splits (not tracked by git)
src/             reusable code (config, data, features, model, evaluate)
notebooks/       experiments and analysis
artifacts/       saved models, scores, metrics
reports/         figures and final report
tests/           sanity checks
```