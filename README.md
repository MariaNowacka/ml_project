# Machine Learning on the S&P 500: Index Predictability and Stock Selection

A machine-learning study of the S&P 500 in two parts:

1. **Index predictability** — can the next-day move of the S&P 500 (SPY ETF) be predicted from the same-day returns of its large constituents?
2. **Stock selection** — can a model pick the individual stocks that will *outperform* SPY over the next month, and can a portfolio of them beat the index?

**Headline results:** the index itself is **not** predictable day-to-day (consistent with weak-form market efficiency), while the stock-selection strategy is evaluated honestly with a walk-forward backtest and SHAP-based interpretation of what drives its picks.

## Part A — Is the index predictable?

Two framings of the same question, one notebook each.

- **Features:** same-day percentage returns of 12 large-cap tickers — AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, BAC, GS, XOM, GLD, JNJ, KO.
- **Target:** the **next-day** SPY return, created with `shift(-1)` so the model always predicts the future from the present — no look-ahead leakage.
- **Period:** daily data from 2005-02-25 onward.

| Notebook | Task | Target | Metric | Baseline |
|---|---|---|---|---|
| `notebooks/ML_methods_price_pred.ipynb` | Regression | next-day SPY return | MAE | predict 0 / predict train mean |
| `notebooks/ML_methods_up_pred.ipynb` | Classification | next-day up/down | Accuracy | always predict "up" |

**Models:** XGBoost, LightGBM (cross-check against a second library), and a Transformer that reads a 10-day window of returns with a learned positional encoding.

**Methodology:**

- **Walk-forward validation** (`TimeSeriesSplit`, 5 folds) — always train on the past, test on the future. Never shuffled.
- **Leakage control** — scaling and sequence-building are fitted inside each fold on its training part only.
- **Fair baselines** — computed within each fold, so comparisons hold even as volatility changes between periods.
- **Tuning** — Optuna (classification), optimizing *accuracy − baseline* (the edge over "always up") rather than raw accuracy.
- **Tracking** — MLflow logs parameters, per-fold metrics, and Optuna trials as nested runs.

**Results:** no model beats the trivial baseline. In regression the trees collapse to the mean and the Transformer adds movement but no direction; in classification the confusion matrices show every model defaulting to the majority class ("up"). Next-day SPY movement is not predictable from same-day constituent returns — the limitation is the data, not the models.

## Part B — A stock-selection strategy

`notebooks/code_final.ipynb` flips the question: instead of forecasting the index, it predicts **which stocks will outperform SPY over the next 22 trading days**, builds a portfolio of the predicted out-performers, and backtests it.

- **Universe:** 27 large-caps mapped to their sector ETFs (XLK, XLV, XLF, XLC, XLY, XLP).
- **Target:** binary — does a stock's forward 22-day return exceed SPY's over the same window?
- **Backtest:** walk-forward with 22-day rebalancing from 2017, portfolio cumulative return compared against SPY, with **alpha** (excess return over SPY) as the headline metric.

Two models are compared:

- **Model 1 — ARIMA + GARCH + LightGBM.** An ARIMA forecast of relative performance and a GARCH volatility estimate are fed, alongside distance to the 200-day moving average, 20-day momentum, and local volatility, into a LightGBM classifier. SHAP shows the ARIMA signal dominating, with GARCH volatility a distant second.
- **Model 2 — LightGBM on engineered features only.** Momentum (5/10/20/60-day), volatility (20/60-day), momentum relative to SPY and to the sector ETF, volume ratios, and calendar features — no ARIMA/GARCH. SHAP shows momentum (especially 60/5/10-day) and volatility carrying most of the weight.

**Interpretability:** SHAP `TreeExplainer` is used throughout — global bar importance, beeswarm plots for the direction of each feature's effect, and per-stock waterfall plots (e.g. TSLA for Model 1, NVDA for Model 2) showing how each feature pushes a single prediction up or down. The final cell plots both strategies against the SPY benchmark and reports each model's alpha.

## Project structure

```
.
├── notebooks/
│   ├── ML_methods_price_pred.ipynb   # Part A: regression — next-day SPY return
│   ├── ML_methods_up_pred.ipynb      # Part A: classification — next-day direction (Optuna, MLflow)
│   └── code_final.ipynb              # Part B: stock-selection strategy (ARIMA+GARCH+LightGBM vs LightGBM, SHAP)
├── Data_clean/
│   ├── Stocks clean/                 # one cleaned CSV per ticker: <ticker>_clean.csv
│   └── ETF clean/                    # SPY and sector ETFs, e.g. spy_clean.csv
├── src/                              # supporting code
├── requirements.txt                  # Python dependencies
└── README.md
```

Each cleaned CSV is indexed by `Date` and contains `Close` (Part B also uses `Volume`).

## Setup

Requires Python 3.12. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

XGBoost needs the OpenMP runtime, which is not a Python package. On macOS install it once with Homebrew:

```bash
brew install libomp
```

## Running

The Part A notebooks read data with relative paths (`../Data_clean/...`), so run them from inside the `notebooks/` folder:

```bash
jupyter lab notebooks/
```

Browse logged experiments and Optuna trials in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000`.

## Key takeaway

The index is efficient on a daily horizon — none of the models in Part A beat a trivial baseline. Part B shifts from predicting the index to selecting stocks that outperform it, and uses SHAP to make the resulting models interpretable rather than black boxes.