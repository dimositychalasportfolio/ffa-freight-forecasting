"""Checks that the report states the numbers it was given, and escapes text safely."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report  # noqa: E402


def walk_summary() -> pd.DataFrame:
    rows = []
    for vessel, period, logistic, low, naive in [
        ("Cape", "4M", 0.64, 0.57, 0.58),
        ("Cape", "Q1", 0.56, 0.47, 0.42),
        ("Panamax", "4M", 0.57, 0.48, 0.42),
    ]:
        rows.append({"Vessel": vessel, "Period": period, "Model": "Logistic Regression",
                     "Pooled_Balanced_Accuracy": logistic, "Pooled_CI95_Low": low,
                     "Pooled_CI95_High": logistic + 0.07, "Pooled_Predictions": 2268})
        rows.append({"Vessel": vessel, "Period": period, "Model": "Majority Baseline",
                     "Pooled_Balanced_Accuracy": naive, "Pooled_CI95_Low": naive - 0.08,
                     "Pooled_CI95_High": naive + 0.08, "Pooled_Predictions": 2268})
    return pd.DataFrame(rows)


def test_findings_name_the_best_result_and_its_baseline():
    findings = report.key_findings(pd.DataFrame(), walk_summary(), pd.DataFrame())
    assert "Capesize +4MON" in findings[0]
    assert "0.64" in findings[0] and "0.57-0.71" in findings[0]
    assert "scores 0.58" in findings[0]


def test_findings_list_only_intervals_above_one_half():
    findings = report.key_findings(pd.DataFrame(), walk_summary(), pd.DataFrame())
    assert "Capesize +4MON" in findings[1]
    assert "Capesize +1Q" not in findings[1]
    assert "Panamax +4MON" not in findings[1]


def test_findings_flag_stale_forecasts():
    live = pd.DataFrame({"Signal_Usable": [False], "As_Of": ["2026-04-21"], "Data_Age_Days": [205]})
    findings = report.key_findings(pd.DataFrame(), walk_summary(), live)
    assert any("not a trading signal" in f for f in findings)


def test_html_table_formats_numbers_and_escapes_text():
    frame = pd.DataFrame({"Name": ["<script>"], "Score": [0.6412]})
    table = report.html_table(frame, {"Name": "Name", "Score": "Score"}, {"Score": ".2f"}, shade=("Score",))
    assert "&lt;script&gt;" in table and "<script>" not in table
    assert "0.64" in table
    assert "rgba(42,120,214" in table  # above 0.50 is shaded blue


def test_experiment_labels_are_readable():
    assert report.experiment_label("Cape", "4M") == "Capesize +4MON"
    assert report.experiment_label("Panamax", "Q2") == "Panamax +2Q"
