# Forecasting Dry-Bulk Freight Futures (FFA): Direction Classification with Leakage-Aware Evaluation

A small research project that tests whether simple technical features contain out-of-sample
information about the direction of Forward Freight Agreement (FFA) prices for Capesize and
Panamax vessels at quarterly and four-month horizons.

**Short answer:** a modest, plausible signal at the +4MON horizon (test balanced accuracy of about
65-72% versus a 50% baseline, with wide confidence intervals). At the +1Q to +3Q horizons the evidence
is weak or absent.
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

- **Features (6):** 1/5/20-day returns, 5/20-day moving-average gap, 20-day volatility, RSI(14). All are
  relative measures; the raw price level is excluded because it is non-stationary (see below).
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

| Vessel | Horizon | Model | Validation bal. acc. | **Test bal. acc.** | Test 95% CI (block bootstrap) |
|---|---|---|---|---|---|
| Capesize | +1Q | Logistic Regression | 0.61 | 0.66 | 0.53 - 0.75 |
| Capesize | +2Q | Gradient Boosting | 0.56 | 0.47 | 0.35 - 0.59 |
| Capesize | +3Q | Logistic Regression | 0.56 | 0.49 | 0.35 - 0.59 |
| Capesize | **+4MON** | Logistic Regression | 0.64 | **0.72** | 0.57 - 0.83 |
| Panamax | +1Q | Logistic Regression | 0.74 | 0.55 | 0.43 - 0.66 |
| Panamax | +2Q | Majority baseline (no model beat it on validation) | 0.50 | 0.50 | - |
| Panamax | +3Q | Extra Trees | 0.64 | 0.59 | 0.51 - 0.67 |
| Panamax | **+4MON** | Logistic Regression | 0.78 | **0.65** | 0.49 - 0.75 |

The intervals use a moving-block bootstrap with block length equal to the forecast horizon, because
overlapping labels are strongly autocorrelated and an ordinary bootstrap would make them far too narrow.

![Model comparison](results/FFA_Model_Accuracy_Comparison.png)

Full per-model results are in [`results/FFA_Model_Comparison.csv`](results/FFA_Model_Comparison.csv); backtest and data-quality summaries are in the same folder.

### What the results say

1. **The only consistent out-of-sample edge is at the 4-month horizon.** Every non-baseline model scores
   0.64-0.73 on the test set for both vessel classes. The Capesize interval stays above 50% (0.57-0.83);
   the Panamax interval just touches it (0.49-0.75). The edge is plausible, not proven.
2. **At quarterly horizons the signal is unreliable.** Most intervals include 50%, and for Panamax +2Q
   no model beat the majority baseline on validation, so the baseline was deployed.
3. **Removing the raw price level removed the validation overfitting.** With the price level as a feature,
   validation balanced accuracy was on average 0.13 higher than test (e.g. Panamax +2Q: 0.90 vs 0.37).
   Without it, validation and test are on average within 0.03 of each other, while average test accuracy
   is unchanged (0.58 vs 0.59). The price level was letting models learn levels specific to one period.

## Limitations (read before citing any number)

- **Backtest figures are not reliable evidence.** Each test backtest contains only 4-10 trades. The Sharpe
  ratio is annualised with sqrt(252 / horizon), i.e. one trade per horizon (an earlier version used sqrt(252),
  which overstated it by a factor of about 8-9). With so few trades, the PnL and Sharpe values are not
  reported here as results.
- **Small effective sample.** Overlapping 63/84-day labels are strongly autocorrelated, so the number of
  independent observations is far lower than the row count.
- Supramax was not available. Live forecasts in the generated report are marked STALE (data ends 2026-04-21)
  and should be ignored.

## Next steps

- Replace the single split with walk-forward evaluation.
- Test term-structure features (spreads between tenors, curve slope).
- Add Supramax and refresh the data.

## Authors

- [Ioannis Katsiris](https://www.linkedin.com/in/ioannis-katsiris-57b7581b6/)
- [Dimos Tychalas](https://www.linkedin.com/in/dimos-tychalas-2757671aa/)

This project was developed jointly.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Outputs: `FFA_Period_Report.txt`, CSV summaries and `FFA_Model_Accuracy_Comparison.png`, written to the project folder.
