"""Vessel- and period-specific FFA forecasting with fair model comparison.

The experiment keeps each vessel class separate and evaluates quarterly FFA
contracts (+1Q, +2Q, +3Q) plus a four-month horizon (+4MON).  Every candidate
model receives the same chronological train/validation/test partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, precision_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
PERIOD_EXPERIMENTS = {
    "Q1": {"tenor": "+1Q", "trading_days": 63, "description": "next quarterly contract"},
    "Q2": {"tenor": "+2Q", "trading_days": 63, "description": "second quarterly contract"},
    "Q3": {"tenor": "+3Q", "trading_days": 63, "description": "third quarterly contract"},
    "4M": {"tenor": "+4MON", "trading_days": 84, "description": "fourth-month contract"},
}
DESIRED_VESSELS = ("Cape", "Panamax", "Supramax")
TRANSACTION_COST_POINTS = 50.0
RANDOM_STATE = 42
MAX_LIVE_DATA_STALENESS_DAYS = 7


@dataclass
class VesselSource:
    vessel: str
    path: Path
    route_prefix: str
    raw_rows: int
    rejected_rows: int


def output_path(filename: str) -> Path:
    return BASE_DIR / filename


def classify_vessel(path: Path, frame: pd.DataFrame) -> str | None:
    """Infer vessel class from the Baltic filename and available group fields."""
    searchable = path.stem.lower()
    if "FFA Group" in frame.columns:
        searchable += " " + " ".join(frame["FFA Group"].dropna().astype(str).str.lower().unique())

    if "cape" in searchable:
        return "Cape"
    if "panamax" in searchable or "pana" in searchable:
        return "Panamax"
    if "supramax" in searchable or "supra" in searchable:
        return "Supramax"
    return None


def infer_route_prefix(periods: pd.Series) -> str | None:
    """Extract the contract prefix, for example ``5TC_C`` from ``5TC_C+1MON``."""
    as_text = periods.dropna().astype(str)
    candidates = as_text[as_text.str.endswith("CURMON")]
    if candidates.empty:
        candidates = as_text[as_text.str.endswith("+1MON")]
        suffix = "+1MON"
    else:
        suffix = "CURMON"

    prefixes = candidates.str.removesuffix(suffix).unique()
    return str(prefixes[0]) if len(prefixes) == 1 else None


def discover_sources() -> tuple[dict[str, VesselSource], list[dict[str, object]]]:
    """Discover valid vessel exports and produce a concise ingestion profile."""
    sources: dict[str, VesselSource] = {}
    quality_rows: list[dict[str, object]] = []

    for path in sorted(BASE_DIR.glob("*.xlsx")):
        try:
            raw = pd.read_excel(path)
        except Exception as exc:
            quality_rows.append(
                {"File": path.name, "Vessel": "Unknown", "Status": f"Unreadable: {exc}", "Raw_Rows": 0}
            )
            continue

        required = {"Date", "Value", "Route-Period"}
        if not required.issubset(raw.columns):
            quality_rows.append(
                {
                    "File": path.name,
                    "Vessel": "Unknown",
                    "Status": "Skipped: missing Date, Value, or Route-Period",
                    "Raw_Rows": len(raw),
                }
            )
            continue

        vessel = classify_vessel(path, raw)
        prefix = infer_route_prefix(raw["Route-Period"])
        if vessel is None or prefix is None:
            quality_rows.append(
                {
                    "File": path.name,
                    "Vessel": vessel or "Unknown",
                    "Status": "Skipped: vessel or route prefix could not be inferred",
                    "Raw_Rows": len(raw),
                }
            )
            continue

        clean_dates = pd.to_datetime(raw["Date"], errors="coerce")
        clean_values = pd.to_numeric(raw["Value"], errors="coerce")
        usable = raw.assign(Date=clean_dates, Value=clean_values).dropna(subset=["Date", "Value", "Route-Period"])
        rejected_rows = len(raw) - len(usable)
        sources[vessel] = VesselSource(vessel, path, prefix, len(raw), rejected_rows)

        for label, config in PERIOD_EXPERIMENTS.items():
            period = f"{prefix}{config['tenor']}"
            series = usable.loc[usable["Route-Period"].astype(str).eq(period), ["Date", "Value"]]
            duplicate_dates = int(series.duplicated(subset="Date").sum())
            series = series.drop_duplicates(subset="Date", keep="last").sort_values("Date")
            quality_rows.append(
                {
                    "File": path.name,
                    "Vessel": vessel,
                    "Period": label,
                    "Contract": period,
                    "Status": "Ready" if not series.empty else "Missing contract",
                    "Raw_Rows": len(raw),
                    "Rejected_Metadata_or_Invalid_Rows": rejected_rows,
                    "Valid_Observations": len(series),
                    "Duplicate_Dates_Removed": duplicate_dates,
                    "Start": series["Date"].min() if not series.empty else pd.NaT,
                    "End": series["Date"].max() if not series.empty else pd.NaT,
                }
            )

    return sources, quality_rows


def read_contract_series(source: VesselSource, tenor: str) -> pd.Series:
    """Return one clean daily price series for a vessel and FFA contract tenor."""
    raw = pd.read_excel(source.path)
    clean = raw.assign(
        Date=pd.to_datetime(raw["Date"], errors="coerce"),
        Value=pd.to_numeric(raw["Value"], errors="coerce"),
    ).dropna(subset=["Date", "Value", "Route-Period"])
    contract = f"{source.route_prefix}{tenor}"
    series = clean.loc[clean["Route-Period"].astype(str).eq(contract), ["Date", "Value"]]
    series = series.drop_duplicates(subset="Date", keep="last").sort_values("Date")
    if series.empty:
        raise ValueError(f"{source.vessel} has no usable {tenor} contract observations.")
    return pd.Series(series["Value"].to_numpy(), index=pd.DatetimeIndex(series["Date"]), name="Price")


def make_feature_frame(price: pd.Series, trading_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build only information available on each decision date plus future labels."""
    frame = pd.DataFrame({"Price": price.astype(float)}).sort_index()
    frame["Return_1D"] = frame["Price"].pct_change(fill_method=None)
    frame["Return_5D"] = frame["Price"].pct_change(5, fill_method=None)
    frame["Return_20D"] = frame["Price"].pct_change(20, fill_method=None)
    frame["MA_Gap"] = frame["Price"].rolling(5).mean() / frame["Price"].rolling(20).mean() - 1
    frame["Volatility_20D"] = frame["Return_1D"].rolling(20).std()

    delta = frame["Price"].diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gains / losses
    frame["RSI_14"] = (100 - (100 / (1 + relative_strength))).replace([np.inf, -np.inf], np.nan)

    frame["Future_Price"] = frame["Price"].shift(-trading_days)
    frame["Future_Return"] = frame["Future_Price"] / frame["Price"] - 1
    frame["Target"] = (frame["Future_Return"] > 0).astype("float")

    features = ["Price", "Return_1D", "Return_5D", "Return_20D", "MA_Gap", "Volatility_20D", "RSI_14"]
    live_frame = frame.dropna(subset=features).copy()
    model_frame = live_frame.dropna(subset=["Future_Price", "Future_Return"]).copy()
    model_frame["Target"] = model_frame["Target"].astype(int)
    return model_frame, live_frame


def chronological_split(
    frame: pd.DataFrame, trading_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series while purging observations whose labels have not matured.

    A row's target uses the price ``trading_days`` ahead.  The two gaps ensure
    that no training or validation target reaches into the next evaluation
    period, which would otherwise make an apparent forecast use future prices.
    """
    validation_start = int(len(frame) * 0.60)
    test_start = int(len(frame) * 0.80)
    train_end = validation_start - trading_days
    validation_end = test_start - trading_days
    train = frame.iloc[:train_end]
    validation = frame.iloc[validation_start:validation_end]
    test = frame.iloc[test_start:]
    if min(len(train), len(validation), len(test)) < 60:
        raise ValueError("Insufficient data for a 60% / 20% / 20% chronological split.")
    return train, validation, test


def classifier_candidates() -> dict[str, object]:
    """Return the same deterministic model families for every vessel/period."""
    gb_grid = GridSearchCV(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.08],
            "model__max_depth": [2],
        },
        cv=TimeSeriesSplit(n_splits=3, gap=84),
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE))]
        ),
        "Random Forest": Pipeline(
            [("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]
        ),
        "Extra Trees": Pipeline(
            [("scaler", StandardScaler()), ("model", ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE))]
        ),
        "Tuned Gradient Boosting": gb_grid,
    }


def non_overlapping_backtest(
    test: pd.DataFrame, probabilities: np.ndarray, threshold: float, trading_days: int
) -> pd.DataFrame:
    """Evaluate horizon trades without claiming that overlapping capital is free."""
    decisions = test.iloc[::trading_days].copy()
    decisions["Probability_Up"] = pd.Series(probabilities, index=test.index).reindex(decisions.index)
    decisions["Signal"] = np.select(
        [decisions["Probability_Up"] > threshold, decisions["Probability_Up"] < (1 - threshold)],
        [1, -1],
        default=0,
    )
    decisions["Size"] = np.where(decisions["Signal"] != 0, np.abs(decisions["Probability_Up"] - 0.5) * 2, 0.0)
    decisions["Gross_PnL"] = decisions["Signal"] * decisions["Size"] * (decisions["Future_Price"] - decisions["Price"])
    decisions["Cost"] = np.where(decisions["Signal"] != 0, 2 * TRANSACTION_COST_POINTS * decisions["Size"], 0.0)
    decisions["Net_PnL"] = decisions["Gross_PnL"] - decisions["Cost"]
    decisions["Cumulative_PnL"] = decisions["Net_PnL"].cumsum()
    return decisions


def sharpe_ratio(pnl: pd.Series, trading_days: int) -> float:
    """Annualised Sharpe ratio for one trade every `trading_days` trading days.

    Trades are non-overlapping and spaced by the forecast horizon, so a year
    holds about 252 / trading_days trades, not 252.
    """
    std = pnl.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(pnl.mean() / std * np.sqrt(252 / trading_days))


BOOTSTRAP_SAMPLES = 2_000


def block_bootstrap_ci(
    y_true: pd.Series, y_pred: np.ndarray, block_length: int, samples: int = BOOTSTRAP_SAMPLES, level: float = 0.95
) -> tuple[float, float]:
    """Moving-block bootstrap interval for balanced accuracy.

    Labels look `block_length` days ahead, so neighbouring rows are strongly
    autocorrelated. Resampling whole blocks keeps that dependence; an ordinary
    row bootstrap would give intervals that are far too narrow.
    """
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    n = len(truth)
    block_length = max(1, min(block_length, n))
    starts = np.arange(n - block_length + 1)
    blocks_needed = int(np.ceil(n / block_length))
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        index = (chosen[:, None] + np.arange(block_length)).ravel()[:n]
        if len(np.unique(truth[index])) < 2:
            continue
        scores.append(balanced_accuracy_score(truth[index], prediction[index]))
    if not scores:
        return float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    return float(np.percentile(scores, tail)), float(np.percentile(scores, 100 - tail))


def choose_threshold(validation: pd.DataFrame, probabilities: np.ndarray, trading_days: int) -> tuple[float, float]:
    candidates = []
    for threshold in np.arange(0.50, 0.66, 0.025):
        backtest = non_overlapping_backtest(validation, probabilities, float(threshold), trading_days)
        candidates.append((float(threshold), sharpe_ratio(backtest["Net_PnL"], trading_days)))

    valid = [(threshold, score) for threshold, score in candidates if np.isfinite(score)]
    return max(valid, key=lambda item: item[1]) if valid else (0.55, float("nan"))


def run_experiment(
    vessel: str, label: str, source: VesselSource
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Compare every classifier fairly, then prepare a validation-selected live model."""
    config = PERIOD_EXPERIMENTS[label]
    trading_days = int(config["trading_days"])
    price = read_contract_series(source, str(config["tenor"]))
    model_frame, live_frame = make_feature_frame(price, trading_days)
    train, validation, test = chronological_split(model_frame, trading_days)
    features = ["Price", "Return_1D", "Return_5D", "Return_20D", "MA_Gap", "Volatility_20D", "RSI_14"]

    if train["Target"].nunique() < 2:
        raise ValueError("Training target has only one class.")

    fitted_models: dict[str, object] = {}
    validation_rows = []
    for name, candidate in classifier_candidates().items():
        candidate.fit(train[features], train["Target"])
        fitted_models[name] = candidate
        prediction = candidate.predict(validation[features])
        validation_rows.append(
            {
                "Model": name,
                "Validation_Accuracy": accuracy_score(validation["Target"], prediction),
                "Validation_Balanced_Accuracy": balanced_accuracy_score(validation["Target"], prediction),
                "Validation_Precision": precision_score(validation["Target"], prediction, zero_division=0),
            }
        )

    validation_scores = pd.DataFrame(validation_rows).sort_values(
        ["Validation_Balanced_Accuracy", "Validation_Accuracy", "Validation_Precision"], ascending=False
    )
    deployment_model_name = str(validation_scores.iloc[0]["Model"])
    deployment_train_model = fitted_models[deployment_model_name]
    validation_probability = deployment_train_model.predict_proba(validation[features])[:, 1]
    threshold, validation_sharpe = choose_threshold(validation, validation_probability, trading_days)

    development = pd.concat([train, validation]).sort_index()
    final_models: dict[str, object] = {}
    model_comparison_rows: list[dict[str, object]] = []
    for name, train_fitted_model in fitted_models.items():
        # Gradient-boosting parameters are tuned only inside the training window.
        # Reusing that fitted pipeline as a template avoids re-tuning on the
        # validation period while still refitting every model on development data.
        final_template = getattr(train_fitted_model, "best_estimator_", train_fitted_model)
        final_model = clone(final_template).fit(development[features], development["Target"])
        final_models[name] = final_model
        test_prediction = final_model.predict(test[features])
        ci_low, ci_high = block_bootstrap_ci(test["Target"], test_prediction, trading_days)
        validation_metrics = validation_scores.loc[validation_scores["Model"].eq(name)].iloc[0]
        model_comparison_rows.append(
            {
                "Vessel": vessel,
                "Period": label,
                "Contract": config["tenor"],
                "Forecast_Horizon_Trading_Days": trading_days,
                "Model": name,
                "Validation_Accuracy": float(validation_metrics["Validation_Accuracy"]),
                "Validation_Balanced_Accuracy": float(validation_metrics["Validation_Balanced_Accuracy"]),
                "Validation_Precision": float(validation_metrics["Validation_Precision"]),
                "Test_Accuracy": accuracy_score(test["Target"], test_prediction),
                "Test_Balanced_Accuracy": balanced_accuracy_score(test["Target"], test_prediction),
                "Test_Balanced_Accuracy_CI95_Low": ci_low,
                "Test_Balanced_Accuracy_CI95_High": ci_high,
                "Test_Precision": precision_score(test["Target"], test_prediction, zero_division=0),
                "Deployment_Model": name == deployment_model_name,
            }
        )

    deployment_classifier = final_models[deployment_model_name]
    test_probability = deployment_classifier.predict_proba(test[features])[:, 1]
    test_prediction = deployment_classifier.predict(test[features])
    final_regressor = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, random_state=RANDOM_STATE)),
        ]
    ).fit(development[features], development["Future_Return"])
    test_return_prediction = final_regressor.predict(test[features])
    backtest = non_overlapping_backtest(test, test_probability, threshold, trading_days)

    comparison_df = pd.DataFrame(model_comparison_rows)
    retrospective_winner = comparison_df.sort_values(
        ["Test_Balanced_Accuracy", "Test_Accuracy", "Test_Precision"], ascending=False
    ).iloc[0]

    latest = live_frame.iloc[[-1]]
    live_classifier = clone(deployment_classifier).fit(model_frame[features], model_frame["Target"])
    live_regressor = clone(final_regressor).fit(model_frame[features], model_frame["Future_Return"])
    probability_up = float(live_classifier.predict_proba(latest[features])[0, 1])
    predicted_return = float(live_regressor.predict(latest[features])[0])
    current_price = float(latest["Price"].iloc[0])
    as_of_date = latest.index[0].date()
    data_age_days = max(0, (pd.Timestamp.today().date() - as_of_date).days)

    if data_age_days > MAX_LIVE_DATA_STALENESS_DAYS:
        signal = "STALE DATA - DO NOT TRADE"
    elif probability_up > threshold:
        signal = "BUY"
    elif probability_up < 1 - threshold:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    deployment_test = comparison_df.loc[comparison_df["Deployment_Model"]].iloc[0]
    summary = {
        "Vessel": vessel,
        "Period": label,
        "Contract": config["tenor"],
        "Period_Description": config["description"],
        "Forecast_Horizon_Trading_Days": trading_days,
        "Observations": len(model_frame),
        "Train_Observations": len(train),
        "Validation_Observations": len(validation),
        "Test_Observations": len(test),
        "Deployment_Classifier": deployment_model_name,
        "Deployment_Validation_Accuracy": float(deployment_test["Validation_Accuracy"]),
        "Deployment_Test_Accuracy": float(deployment_test["Test_Accuracy"]),
        "Deployment_Test_Balanced_Accuracy": float(deployment_test["Test_Balanced_Accuracy"]),
        "Deployment_Test_Balanced_Accuracy_CI95_Low": float(deployment_test["Test_Balanced_Accuracy_CI95_Low"]),
        "Deployment_Test_Balanced_Accuracy_CI95_High": float(deployment_test["Test_Balanced_Accuracy_CI95_High"]),
        "Best_Test_Classifier": str(retrospective_winner["Model"]),
        "Best_Test_Accuracy": float(retrospective_winner["Test_Accuracy"]),
        "Threshold": threshold,
        "Validation_Sharpe": validation_sharpe,
        "Test_MAE_Future_Return": mean_absolute_error(test["Future_Return"], test_return_prediction),
        "Test_Trade_Count": int((backtest["Signal"] != 0).sum()),
        "Test_Net_PnL_Points": float(backtest["Net_PnL"].sum()),
        "Test_Sharpe": sharpe_ratio(backtest["Net_PnL"], trading_days),
        "Last_Data_Date": latest.index[0].date().isoformat(),
    }
    live = {
        "Vessel": vessel,
        "Period": label,
        "Contract": config["tenor"],
        "Forecast_Horizon_Trading_Days": trading_days,
        "As_Of": as_of_date.isoformat(),
        "Data_Age_Days": data_age_days,
        "Signal_Usable": data_age_days <= MAX_LIVE_DATA_STALENESS_DAYS,
        "Current_FFA_Price": current_price,
        "Predicted_Return": predicted_return,
        "Predicted_Future_Price": current_price * (1 + predicted_return),
        "Probability_Up": probability_up,
        "Probability_Down": 1 - probability_up,
        "Signal": signal,
        "Position_Size": (
            abs(probability_up - 0.5) * 2
            if signal not in {"NEUTRAL", "STALE DATA - DO NOT TRADE"}
            else 0.0
        ),
        "Deployment_Classifier": deployment_model_name,
        "Threshold": threshold,
    }
    return summary, live, model_comparison_rows


def plot_model_accuracy(model_comparison: pd.DataFrame) -> None:
    """Show every model on the same out-of-sample accuracy scale."""
    if model_comparison.empty:
        return

    comparison = model_comparison.copy()
    comparison["Experiment"] = comparison["Vessel"] + " " + comparison["Period"]
    experiment_order = comparison[["Experiment", "Vessel", "Period"]].drop_duplicates().sort_values(["Vessel", "Period"])[
        "Experiment"
    ]
    model_order = [
        "Majority Baseline",
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Tuned Gradient Boosting",
    ]
    pivot = comparison.pivot(index="Experiment", columns="Model", values="Test_Balanced_Accuracy").reindex(experiment_order)
    x = np.arange(len(pivot.index))
    width = 0.16

    plt.figure(figsize=(13, 6))
    for position, model_name in enumerate(model_order):
        if model_name not in pivot:
            continue
        values = pivot[model_name].to_numpy() * 100
        plt.bar(x + (position - 2) * width, values, width, label=model_name)
    plt.axhline(50, color="black", linewidth=1, linestyle="--", label="50% balanced-accuracy baseline")
    plt.ylabel("Out-of-sample balanced accuracy (%)")
    plt.title("FFA Model Comparison by Vessel and Period")
    plt.ylim(0, 100)
    plt.xticks(x, pivot.index, rotation=25, ha="right")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path("FFA_Model_Accuracy_Comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


def format_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    if frame.empty:
        return "No successful experiments."
    table = frame.loc[:, list(columns)].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "n/a")
    return table.to_string(index=False)


def main() -> None:
    print("FFA vessel, quarterly, and four-month forecasting experiment")
    print("Periods: +1Q, +2Q, +3Q (63-day forecast) and +4MON (84-day forecast)")
    sources, quality_rows = discover_sources()
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(output_path("FFA_Data_Quality_Summary.csv"), index=False)

    available_vessels = sorted(sources)
    missing_vessels = [vessel for vessel in DESIRED_VESSELS if vessel not in sources]
    print(f"Available vessel data: {', '.join(available_vessels) or 'none'}")
    if missing_vessels:
        print(f"No source export found for: {', '.join(missing_vessels)}")

    summaries: list[dict[str, object]] = []
    live_forecasts: list[dict[str, object]] = []
    model_comparisons: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for vessel, source in sources.items():
        for label in PERIOD_EXPERIMENTS:
            try:
                print(f"Running {vessel} {label}...")
                summary, live, comparison = run_experiment(vessel, label, source)
                summaries.append(summary)
                live_forecasts.append(live)
                model_comparisons.extend(comparison)
            except Exception as exc:
                failures.append({"Vessel": vessel, "Period": label, "Status": str(exc)})
                print(f"Skipped {vessel} {label}: {exc}")

    summary_df = pd.DataFrame(summaries)
    live_df = pd.DataFrame(live_forecasts)
    model_comparison_df = pd.DataFrame(model_comparisons)
    failures_df = pd.DataFrame(failures, columns=["Vessel", "Period", "Status"])
    summary_df.to_csv(output_path("FFA_Period_Backtest_Summary.csv"), index=False)
    live_df.to_csv(output_path("FFA_Live_Forecasts.csv"), index=False)
    model_comparison_df.to_csv(output_path("FFA_Model_Comparison.csv"), index=False)
    failures_df.to_csv(output_path("FFA_Experiment_Failures.csv"), index=False)
    plot_model_accuracy(model_comparison_df)

    max_data_age = int(live_df["Data_Age_Days"].max()) if not live_df.empty else 0
    live_signal_status = (
        "STALE — refresh the Baltic Exchange exports before using any live signal."
        if max_data_age > MAX_LIVE_DATA_STALENESS_DAYS
        else "Current — live signals are within the configured freshness threshold."
    )

    report = f"""==================================================
FFA PERIOD & VESSEL FORECAST REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

Method
------
Each experiment keeps vessel classes and contract periods separate.  The model
forecasts +1Q, +2Q, and +3Q over 63 trading days (about one quarter), plus
+4MON over 84 trading days (about four months).  The source contains a
fourth-month contract, not a four-month average delivery contract.

Every model uses the same chronological 60% / 20% / 20% split, with a
target-maturity gap equal to the forecast horizon at each boundary.  The
deployment model is chosen only on validation balanced accuracy;
the withheld test set is reported for every candidate model.  Thresholds are
also selected only on validation data.  The PnL check uses non-overlapping
horizon trades.

Data availability
-----------------
Available vessel classes: {', '.join(available_vessels) or 'none'}
Requested but absent source files: {', '.join(missing_vessels) or 'none'}

Deployment-model summary
------------------------
{format_table(summary_df, ['Vessel', 'Period', 'Contract', 'Deployment_Classifier', 'Deployment_Test_Accuracy', 'Deployment_Test_Balanced_Accuracy', 'Deployment_Test_Balanced_Accuracy_CI95_Low', 'Deployment_Test_Balanced_Accuracy_CI95_High', 'Best_Test_Classifier', 'Best_Test_Accuracy', 'Test_Trade_Count', 'Test_Net_PnL_Points', 'Test_Sharpe'])}

Complete model comparison
-------------------------
{format_table(model_comparison_df, ['Vessel', 'Period', 'Model', 'Validation_Accuracy', 'Validation_Balanced_Accuracy', 'Test_Accuracy', 'Test_Balanced_Accuracy', 'Test_Balanced_Accuracy_CI95_Low', 'Test_Balanced_Accuracy_CI95_High', 'Test_Precision', 'Deployment_Model'])}

Current forecasts
-----------------
{format_table(live_df, ['Vessel', 'Period', 'Contract', 'As_Of', 'Data_Age_Days', 'Current_FFA_Price', 'Predicted_Future_Price', 'Probability_Up', 'Signal', 'Position_Size'])}

Live-data status: {live_signal_status}

Caveats
-------
Longer-horizon experiments have fewer non-overlapping trades, so their Sharpe
estimates are less stable.  Sharpe is annualised with sqrt(252 / horizon), one
trade per horizon.  Balanced-accuracy intervals use a moving-block bootstrap
(block = forecast horizon) because overlapping labels are autocorrelated.
Accuracy alone is not enough: balanced accuracy is
reported to prevent an imbalanced up/down class from looking artificially good.
Supramax will be included automatically when its Baltic Exchange Excel export
is added to this directory and identifies the vessel/route period in the same
way as the existing files.
"""
    output_path("FFA_Period_Report.txt").write_text(report, encoding="utf-8")
    print("Complete — summaries, live forecasts, quality checks, and chart saved.")


if __name__ == "__main__":
    main()
