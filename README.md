# Predicting Daily S&P 500 Movements from Constituent Returns

A machine-learning study testing whether the **next-day movement of the S&P 500** (via the SPY ETF) can be predicted from the **same-day returns of 12 large-cap constituents**. The problem is framed two ways — regression (predict the return) and classification (predict up/down) — tackled with three model families and evaluated with walk-forward validation against trivial baselines.

**Headline result:** none of the models beat a trivial baseline. Daily SPY movements are not predictable from same-day constituent returns — a result consistent with the weak form of the efficient-market hypothesis. The limitation is the data, not the models or their hyperparameters.

## Problem setup

- **Features:** same-day percentage returns of 12 large-cap tickers — AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, BAC, GS, XOM, GLD, JNJ, KO.
- **Target:** the **next-day** SPY return, created with `shift(-1)` so the model always predicts the future from the present — no look-ahead leakage.
- **Period:** daily data from 2005-02-25 onward, trimmed to the range where all tickers have data.

## Two tasks

| Notebook | Task | Target | Metric | Baseline |
|---|---|---|---|---|
| `notebooks/ML_methods_price_pred.ipynb` | Regression | next-day SPY return | MAE | predict 0 / predict train mean |
| `notebooks/ML_methods_up_pred.ipynb` | Classification | next-day up/down | Accuracy | always predict "up" |

## Models

Three model families, evaluated identically across both tasks:

- **XGBoost** — gradient-boosted trees, one day of returns at a time.
- **LightGBM** — a second boosting library, used as a cross-check that results are not specific to one implementation.
- **Transformer** — a sequence model that sees a 10-day window of returns (with a learned positional encoding) rather than a single day.

## Methodology

- **Walk-forward validation** (`TimeSeriesSplit`, 5 folds) — always train on the past, test on the future. Data is never shuffled, which would leak future information into training.
- **Leakage control** — scaling and sequence-building are fitted inside each fold on its training part only, then applied to that fold's test part.
- **Fair baselines** — computed within each fold, so models are compared against a same-period reference even as volatility changes between periods.
- **Hyperparameter tuning** — Optuna (classification notebook), optimizing *accuracy − baseline* (the edge over "always up") rather than raw accuracy, so the search isn't rewarded for simply landing on folds with a higher up-share.
- **Experiment tracking** — MLflow logs parameters, per-fold metrics, and each Optuna trial as a nested run.

## Results

- **Regression:** every model's MAE sits at or above the zero-baseline. The tree models collapse to predicting the mean; the Transformer adds movement but no direction, ending up worse than the flat baseline.
- **Classification:** no model beats the "always up" baseline by a meaningful margin. Confusion matrices show the models default to the majority class ("up") rather than detecting direction, and the best Optuna-tuned result is well within noise (std ≈ 10× the mean edge over baseline).

## Project structure

```
.
├── notebooks/
│   ├── ML_methods_price_pred.ipynb   # regression: next-day SPY return
│   └── ML_methods_up_pred.ipynb      # classification: next-day direction (+ Optuna, MLflow)
├── Data_clean/
│   ├── Stocks clean/                 # one cleaned CSV per ticker: <ticker>_clean.csv
│   └── ETF clean/
│       └── spy_clean.csv             # SPY target series
├── src/                              # supporting code
├── requirements.txt                  # Python dependencies
└── README.md
```

Each cleaned CSV is indexed by `Date` and contains a `Close` column, which is what the notebooks read.

## Setup

Requires Python 3.12. From the project root:

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

XGBoost needs the OpenMP runtime, which is not a Python package. On macOS install it once with Homebrew:

```bash
brew install libomp
```

## Running

The notebooks live in `notebooks/` and read the data with relative paths (`../Data_clean/...`), so run them **from inside the `notebooks/` folder** (open the project folder in your editor and launch the notebook there):

```bash
jupyter lab notebooks/
```

To browse the logged experiments and Optuna trials in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000`.

## Key takeaway

Across all three models, a 5-fold walk-forward, and Optuna tuning, next-day SPY movement could not be predicted from same-day constituent returns. A negative result, but a clean one — and exactly what weak-form market efficiency predicts.