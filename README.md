# Forecasting Dry-Bulk Freight Futures (FFA): Direction Classification with Leakage-Aware Evaluation

A small research project that tests whether simple technical features contain out-of-sample
information about the direction of Forward Freight Agreement (FFA) prices for Capesize and
Panamax vessels at quarterly and four-month horizons.

**Short answer:** a modest, plausible signal at the +4MON horizon (test balanced accuracy of about
66-69% versus a 50% baseline). At the +1Q to +3Q horizons the evidence is weak or absent.
The project's main contribution is the evaluation design, not a trading strategy.

## Question

Can price-history features predict whether an FFA contract will be higher or lower
63 trading days (about one quarter) or 84 trading days (about four months) ahead?

## Data

Daily Baltic Exchange FFA history (5TC Capesize from 2014, 5TC Panamax from 2019, to April 2026),
contracts +1Q, +2Q, +3Q and +4MON.

**The data is not included in this repository** (licensed by the Baltic Exchange).
To reproduce, place the two Excel exports in the project folder; `ffa_panel.py` detects the vessel
class from the file name and the contract from the `Route-Period` column.

## Method

- **Features (7):** price level, 1/5/20-day returns, 5/20-day moving-average gap, 20-day volatility, RSI(14).
- **Target:** price higher after the horizon (1) or not (0).
- **Split:** chronological 60% / 20% / 20% (train / validation / test), with a **purge gap equal to the
  forecast horizon** at each boundary so that no label reaches into the next period.
- **Models, identical for every experiment:** majority baseline, logistic regression, random forest,
  extra trees, gradient boosting (grid-searched with `TimeSeriesSplit`).
- **Selection:** deployment model and trading threshold chosen **on validation data only**.
  Test results are reported for every candidate, not just the winner.
- **Metric:** balanced accuracy (the up/down classes are imbalanced in some windows), with a majority
  baseline for reference.
- **Backtest check:** non-overlapping horizon trades, 50-point transaction cost per side, position size
  scaled by model confidence.

## Results (deployment model chosen on validation, evaluated on the untouched test set)

| Vessel | Horizon | Model | Validation bal. acc. | **Test bal. acc.** |
|---|---|---|---|---|
| Capesize | +1Q | Logistic Regression | 0.72 | 0.63 |
| Capesize | +2Q | Logistic Regression | 0.63 | 0.58 |
| Capesize | +3Q | Gradient Boosting | 0.67 | 0.48 |
| Capesize | **+4MON** | Logistic Regression | 0.73 | **0.69** |
| Panamax | +1Q | Gradient Boosting | 0.79 | 0.53 |
| Panamax | +2Q | Logistic Regression | 0.90 | 0.37 |
| Panamax | +3Q | Gradient Boosting | 0.78 | 0.51 |
| Panamax | **+4MON** | Logistic Regression | 0.90 | **0.66** |

Full per-model results are in [`results/FFA_Model_Comparison.csv`](results/FFA_Model_Comparison.csv); backtest and data-quality summaries are in the same folder.

### What the results say

1. **The only consistent out-of-sample edge is at the 4-month horizon**, for both vessel classes and
   for every non-baseline model (balanced accuracy 61-71%).
2. **At quarterly horizons the signal is unreliable.** Two of six deployment models score at or below
   50% on the test set.
3. **Validation scores are much higher than test scores** (e.g. Panamax +2Q: 0.90 vs 0.37). Validation
   accuracy was therefore not a trustworthy guide to out-of-sample performance, which points to regime
   change between the windows and overfitting to the validation period.

## Limitations (read before citing any number)

- **Backtest figures are not reliable evidence.** Each test backtest contains only 4-10 trades, and the
  Sharpe ratio in the code is annualised with sqrt(252) although trades occur every 63-84 days, which
  overstates it. The reported PnL and Sharpe values are therefore not reported here as results.
- **Small effective sample.** Overlapping 63/84-day labels are strongly autocorrelated, so the number of
  independent observations is far lower than the row count.
- **The price level is used as a feature**, which is non-stationary and probably drives some of the
  regime-dependence seen above.
- **No confidence intervals** on the balanced-accuracy figures yet.
- Supramax was not available. Live forecasts in the generated report are marked STALE (data ends 2026-04-21)
  and should be ignored.

## Next steps

- Replace the single split with walk-forward evaluation and add bootstrap confidence intervals.
- Drop the raw price level and test stationary features (spreads between tenors, curve slope).
- Correct the Sharpe annualisation to the actual trade frequency.
- Add Supramax and refresh the data.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Outputs: `FFA_Period_Report.txt`, CSV summaries and `FFA_Model_Accuracy_Comparison.png`, written to the project folder.
