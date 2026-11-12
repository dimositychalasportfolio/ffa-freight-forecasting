"""Charts, an HTML report and an Excel workbook from the experiment results.

Everything here only *presents* results that ffa_panel.py has already computed;
nothing in this file trains a model or changes a number.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # draw to files, not to a window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ffa_panel import ExperimentResult


# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BASELINE_GREY = "#8a8984"

# Categorical colours, always in this order, so a model keeps its colour in every chart.
MODEL_COLOURS = {
    "Logistic Regression": "#2a78d6",
    "Random Forest": "#eb6834",
    "Extra Trees": "#1baf7a",
    "Tuned Gradient Boosting": "#eda100",
    "Majority Baseline": BASELINE_GREY,
}
MODEL_ORDER = list(MODEL_COLOURS)

# Train / validation / test shading: one blue ramp, light to darker.
SPLIT_SHADES = {"Train": "#e6f0fc", "Validation": "#b7d3f6", "Test": "#86b6ef"}
RISE, FALL = "#2a78d6", "#e34948"  # diverging pair for "up" vs "down"
PERIOD_ORDER = ["Q1", "Q2", "Q3", "4M"]
TENOR = {"Q1": "+1Q", "Q2": "+2Q", "Q3": "+3Q", "4M": "+4MON"}

FIGURES_DIR = "figures"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "font.size": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def experiment_label(vessel: str, period: str) -> str:
    return f"{'Capesize' if vessel == 'Cape' else vessel} {TENOR.get(period, period)}"


def ordered_results(results: list["ExperimentResult"]) -> list["ExperimentResult"]:
    """Capesize first, then Panamax; contracts from nearest to furthest."""
    def key(result: "ExperimentResult") -> tuple[int, int]:
        vessel = str(result.summary["Vessel"])
        period = str(result.summary["Period"])
        return (0 if vessel == "Cape" else 1, PERIOD_ORDER.index(period) if period in PERIOD_ORDER else 9)

    return sorted(results, key=key)


def panel_grid(results: list["ExperimentResult"], height: float = 2.6):
    """One row per vessel, one column per contract."""
    vessels = list(dict.fromkeys(str(r.summary["Vessel"]) for r in results))
    fig, axes = plt.subplots(
        len(vessels), len(PERIOD_ORDER), figsize=(14, height * len(vessels) + 0.8), squeeze=False
    )
    positions = {}
    for result in results:
        row = vessels.index(str(result.summary["Vessel"]))
        col = PERIOD_ORDER.index(str(result.summary["Period"]))
        positions[id(result)] = axes[row][col]
    used = set(positions.values())
    for ax in axes.flat:
        if ax not in used:
            ax.set_visible(False)
    return fig, positions


def date_axis(ax) -> None:
    """Readable year/month ticks whether a panel spans two years or twelve."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def save(fig, folder: Path, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return f"{FIGURES_DIR}/{name}"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_prices_and_splits(results: list["ExperimentResult"], folder: Path) -> str:
    """Price history of every contract, shaded by train / validation / test."""
    fig, positions = panel_grid(results)
    for result in results:
        ax = positions[id(result)]
        dates = result.split_dates
        for name, start, end in [
            ("Train", dates["train_start"], dates["train_end"]),
            ("Validation", dates["validation_start"], dates["validation_end"]),
            ("Test", dates["test_start"], dates["test_end"]),
        ]:
            ax.axvspan(start, end, color=SPLIT_SHADES[name], lw=0, label=name, zorder=0)
        ax.plot(result.price.index, result.price.to_numpy(), color=TEXT, lw=0.8, zorder=2)
        ax.set_title(experiment_label(str(result.summary["Vessel"]), str(result.summary["Period"])))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        date_axis(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPLIT_SHADES[n]) for n in SPLIT_SHADES]
    fig.legend(handles, ["Train (60%)", "Validation (20%)", "Test (20%)"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("FFA prices ($/day) and how each series is split in time", y=1.07, fontsize=13,
                 fontweight="bold", color=TEXT)
    fig.text(0.5, -0.02, "White gaps between shaded blocks are the purge gaps (one forecast horizon).",
             ha="center", color=TEXT_MUTED)
    fig.tight_layout()
    return save(fig, folder, "1_prices_and_splits.png")


def chart_test_accuracy(model_comparison: pd.DataFrame, folder: Path) -> str:
    """Every model's test balanced accuracy with its 95% interval."""
    data = model_comparison.copy()
    data["Experiment"] = [experiment_label(v, p) for v, p in zip(data["Vessel"], data["Period"])]
    order = (
        data.assign(v=(data["Vessel"] != "Cape").astype(int), p=data["Period"].map(PERIOD_ORDER.index))
        .sort_values(["v", "p"])["Experiment"].drop_duplicates().tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(order) + 1.4))
    offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER):
        rows = data[data["Model"] == model].set_index("Experiment").reindex(order)
        y = np.arange(len(order)) + offset
        ax.hlines(y, rows["Test_Balanced_Accuracy_CI95_Low"], rows["Test_Balanced_Accuracy_CI95_High"],
                  color=MODEL_COLOURS[model], lw=1.6, alpha=0.8)
        ax.scatter(rows["Test_Balanced_Accuracy"], y, s=34, color=MODEL_COLOURS[model],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3, label=model)
        chosen = rows["Deployment_Model"].fillna(False).astype(bool).to_numpy()
        ax.scatter(rows["Test_Balanced_Accuracy"][chosen], y[chosen], s=110, facecolor="none",
                   edgecolor=TEXT, linewidth=1.2, zorder=4)
    ax.axvline(0.5, color=TEXT_MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlim(0.15, 0.95)
    ax.set_xlabel("Test balanced accuracy (dot) with 95% block-bootstrap interval (line); 0.50 = no skill")
    ax.scatter([], [], s=110, facecolor="none", edgecolor=TEXT, label="Chosen on validation")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_title("Single held-out test period: every model, with uncertainty", pad=46)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, folder, "2_test_accuracy_with_intervals.png")


def chart_walk_forward(results: list["ExperimentResult"], folds: pd.DataFrame, folder: Path) -> str:
    """Balanced accuracy on each six-month walk-forward block, over time."""
    fig, positions = panel_grid(results)
    for result in results:
        ax = positions[id(result)]
        vessel, period = str(result.summary["Vessel"]), str(result.summary["Period"])
        mine = folds[(folds["Vessel"] == vessel) & (folds["Period"] == period)].copy()
        mine["Middle"] = pd.to_datetime(mine["Test_Start"]) + (
            pd.to_datetime(mine["Test_End"]) - pd.to_datetime(mine["Test_Start"])
        ) / 2
        line = mine[mine["Model"] == "Logistic Regression"].sort_values("Middle")
        ax.plot(line["Middle"], line["Balanced_Accuracy"], "-", color=MODEL_COLOURS["Logistic Regression"],
                lw=1.6, marker="o", ms=4.5, label="Logistic regression, one dot per six-month block")
        skipped = mine[(mine["Model"] == "Logistic Regression") & mine["Balanced_Accuracy"].isna()]
        for middle in skipped["Middle"]:
            ax.axvline(middle, color=GRID, lw=5, zorder=0)
        pooled = next((r for r in result.walk_forward_pooled if r["Model"] == "Logistic Regression"), None)
        naive = next((r for r in result.walk_forward_pooled if r["Model"] == "Majority Baseline"), None)
        title = experiment_label(vessel, period)
        if pooled and naive:
            title += (f"\nall blocks: {pooled['Pooled_Balanced_Accuracy']:.2f} "
                      f"({pooled['Pooled_CI95_Low']:.2f}-{pooled['Pooled_CI95_High']:.2f}), "
                      f"naive {naive['Pooled_Balanced_Accuracy']:.2f}")
        ax.set_title(title, fontsize=10)
        ax.axhline(0.5, color=TEXT_MUTED, ls=":", lw=1)
        ax.set_ylim(0, 1.02)
        date_axis(ax)
    handles, labels = next(iter(positions.values())).get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID))
    labels.append("Block with only rises or only falls (not scored)")
    handles.append(plt.Line2D([], [], color=TEXT_MUTED, ls=":"))
    labels.append("0.50 = no skill")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02, "Title: balanced accuracy over all out-of-sample days with 95% interval, and the naive "
             "baseline (predicts the training window's majority direction) on the same days.",
             ha="center", color=TEXT_MUTED)
    fig.suptitle("Walk-forward: balanced accuracy on each six-month test block", y=1.07, fontsize=13,
                 fontweight="bold", color=TEXT)
    fig.tight_layout()
    return save(fig, folder, "3_walk_forward_over_time.png")


def chart_validation_vs_test(model_comparison: pd.DataFrame, folder: Path) -> str:
    """Points far below the diagonal looked better on validation than they really were."""
    fig, ax = plt.subplots(figsize=(6.2, 6))
    others = model_comparison[~model_comparison["Deployment_Model"].astype(bool)]
    chosen = model_comparison[model_comparison["Deployment_Model"].astype(bool)]
    ax.plot([0, 1], [0, 1], color=TEXT_MUTED, lw=1, ls="--")
    ax.scatter(others["Validation_Balanced_Accuracy"], others["Test_Balanced_Accuracy"], s=36,
               color="#86b6ef", edgecolor=SURFACE, linewidth=1, label="Other models")
    ax.scatter(chosen["Validation_Balanced_Accuracy"], chosen["Test_Balanced_Accuracy"], s=60,
               color="#2a78d6", edgecolor=SURFACE, linewidth=1.2, label="Model chosen on validation", zorder=3)
    for _, row in chosen.iterrows():
        ax.annotate(experiment_label(row["Vessel"], row["Period"]),
                    (row["Validation_Balanced_Accuracy"], row["Test_Balanced_Accuracy"]),
                    xytext=(5, -3), textcoords="offset points", fontsize=7.5, color=TEXT_MUTED)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Validation balanced accuracy")
    ax.set_ylabel("Test balanced accuracy")
    ax.text(0.97, 0.33, "below the line:\nlooked better on validation\nthan on test", ha="right", va="bottom",
            color=TEXT_MUTED, fontsize=8)
    ax.legend(loc="upper left")
    ax.set_title("Did validation predict test performance?")
    fig.tight_layout()
    return save(fig, folder, "4_validation_vs_test.png")


def chart_test_predictions(results: list["ExperimentResult"], folder: Path) -> str:
    """The chosen model's probability of a rise through the test period, against what happened."""
    fig, positions = panel_grid(results, height=2.4)
    for result in results:
        ax = positions[id(result)]
        tp = result.test_predictions
        threshold = float(result.summary["Threshold"])
        ax.axhspan(1 - threshold, threshold, color=GRID, lw=0, zorder=0)
        ax.plot(tp.index, tp["Probability_Up"], color=TEXT, lw=1.1, zorder=2)
        up = tp[tp["Actual_Up"] == 1]
        down = tp[tp["Actual_Up"] == 0]
        ax.scatter(up.index, np.full(len(up), 1.04), s=5, color=RISE, marker="|", zorder=3)
        ax.scatter(down.index, np.full(len(down), -0.04), s=5, color=FALL, marker="|", zorder=3)
        hit = (tp["Predicted_Up"] == tp["Actual_Up"]).mean()
        ax.set_title(f"{experiment_label(str(result.summary['Vessel']), str(result.summary['Period']))}\n"
                     f"{result.summary['Deployment_Classifier']} · {hit:.0%} of days correct", fontsize=9.5)
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(0.5, color=TEXT_MUTED, ls=":", lw=0.8)
        date_axis(ax)
    handles = [
        plt.Line2D([], [], color=TEXT, lw=1.1),
        plt.Rectangle((0, 0), 1, 1, color=GRID),
        plt.Line2D([], [], color=RISE, marker="|", ls="", ms=9),
        plt.Line2D([], [], color=FALL, marker="|", ls="", ms=9),
    ]
    fig.legend(handles, ["Predicted probability of a rise", "No-trade band (between thresholds)",
                         "Price actually rose (top)", "Price actually fell (bottom)"],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Test period: what the chosen model predicted, day by day", y=1.07, fontsize=13,
                 fontweight="bold", color=TEXT)
    fig.tight_layout()
    return save(fig, folder, "5_test_predictions.png")


def chart_latest_forecast(live: pd.DataFrame, folder: Path) -> str:
    """Latest probability of a rise for every contract, drawn from the 50% line."""
    data = live.copy()
    data["Experiment"] = [experiment_label(v, p) for v, p in zip(data["Vessel"], data["Period"])]
    data = data.assign(v=(data["Vessel"] != "Cape").astype(int), p=data["Period"].map(PERIOD_ORDER.index))
    data = data.sort_values(["v", "p"])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1.6))
    y = np.arange(len(data))
    probability = data["Probability_Up"].to_numpy()
    colours = [RISE if p >= 0.5 else FALL for p in probability]
    ax.barh(y, probability - 0.5, left=0.5, color=colours, height=0.55)
    ax.set_axisbelow(True)
    for yi, p, model in zip(y, probability, data["Deployment_Classifier"]):
        note = "  (naive baseline: always the majority direction)" if model == "Majority Baseline" else ""
        inside = p > 0.8
        ax.text(p - 0.012 if inside else p + (0.012 if p >= 0.5 else -0.012), yi, f"{p:.0%}{note}", va="center",
                ha="right" if inside or p < 0.5 else "left", color=SURFACE if inside else TEXT, fontsize=8.5)
    ax.axvline(0.5, color=TEXT, lw=1)
    ax.set_yticks(y, data["Experiment"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Model probability that the price is higher at the horizon  (left: fall, right: rise)")
    as_of = data["As_Of"].max()
    stale = not bool(data["Signal_Usable"].all())
    ax.set_title(f"Latest forecast, data as of {as_of}" + ("  -  STALE DATA, DO NOT TRADE" if stale else ""))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, folder, "6_latest_forecast.png")


# ---------------------------------------------------------------------------
# Plain-language findings, computed from the results (never typed in by hand)
# ---------------------------------------------------------------------------

def key_findings(model_comparison: pd.DataFrame, walk_summary: pd.DataFrame, live: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    logistic = walk_summary[walk_summary["Model"] == "Logistic Regression"].copy()
    baseline = walk_summary[walk_summary["Model"] == "Majority Baseline"].set_index(["Vessel", "Period"])
    if not logistic.empty:
        best = logistic.sort_values("Pooled_Balanced_Accuracy", ascending=False).iloc[0]
        base = baseline.loc[(best["Vessel"], best["Period"]), "Pooled_Balanced_Accuracy"]
        findings.append(
            f"Strongest walk-forward result: {experiment_label(best['Vessel'], best['Period'])}, logistic "
            f"regression balanced accuracy {best['Pooled_Balanced_Accuracy']:.2f} (95% CI "
            f"{best['Pooled_CI95_Low']:.2f}-{best['Pooled_CI95_High']:.2f}) over {int(best['Pooled_Predictions']):,} "
            f"out-of-sample days. A naive baseline that predicts the recent majority direction scores {base:.2f}."
        )
        clear = logistic[logistic["Pooled_CI95_Low"] > 0.5]
        names = ", ".join(experiment_label(v, p) for v, p in zip(clear["Vessel"], clear["Period"])) or "none"
        findings.append(
            f"Experiments where the logistic-regression interval stays above 0.50 in walk-forward: {names} "
            f"(out of {len(logistic)})."
        )
    if not model_comparison.empty:
        ml = model_comparison[model_comparison["Model"] != "Majority Baseline"]
        gap = (ml["Validation_Balanced_Accuracy"] - ml["Test_Balanced_Accuracy"]).mean()
        findings.append(
            f"Average validation minus test balanced accuracy across models: {gap:+.2f}. "
            "Close to zero means validation was an honest guide to test performance."
        )
    if not live.empty and not bool(live["Signal_Usable"].all()):
        findings.append(
            f"Latest forecasts use data up to {live['As_Of'].max()} ({int(live['Data_Age_Days'].max())} days old): "
            "treat them as an illustration, not a trading signal."
        )
    return findings


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def html_table(frame: pd.DataFrame, columns: dict[str, str], formats: dict[str, str] | None = None,
               shade: tuple[str, ...] = ()) -> str:
    """A small, styled HTML table. `shade` columns get a soft colour around 0.50."""
    formats = formats or {}
    head = "".join(f"<th>{html.escape(label)}</th>" for label in columns.values())
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                text = "-"
            elif column in formats:
                text = format(value, formats[column])
            else:
                text = str(value)
            style = ""
            if column in shade and pd.notna(value):
                strength = min(abs(float(value) - 0.5) / 0.25, 1.0)
                colour = "42,120,214" if float(value) >= 0.5 else "227,73,72"
                style = f' style="background: rgba({colour},{0.08 + 0.32 * strength:.2f})"'
            numeric = isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
            cells.append(f'<td class="{"num" if numeric else ""}"{style}>{html.escape(text)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def embed_image(output_dir: Path, relative_path: str, alt: str, narrow: bool = False) -> str:
    data = base64.b64encode((output_dir / relative_path).read_bytes()).decode("ascii")
    css_class = ' class="narrow"' if narrow else ""
    return f'<figure{css_class}><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"></figure>'


def build_html(
    output_dir: Path,
    figures: dict[str, str],
    summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    walk_summary: pd.DataFrame,
    live: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    label = lambda frame: [experiment_label(v, p) for v, p in zip(frame["Vessel"], frame["Period"])]  # noqa: E731

    def in_order(frame: pd.DataFrame) -> pd.DataFrame:
        """Capesize before Panamax, nearest contract first: the same order as the charts."""
        keys = frame.assign(_v=(frame["Vessel"] != "Cape").astype(int),
                            _p=frame["Period"].map(lambda p: PERIOD_ORDER.index(p) if p in PERIOD_ORDER else 9))
        sort_by = ["_v", "_p"] + (["_m"] if "Model" in frame else [])
        if "Model" in frame:
            keys["_m"] = keys["Model"].map(lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 9)
        return keys.sort_values(sort_by).drop(columns=[c for c in ("_v", "_p", "_m") if c in keys])

    summary, model_comparison, walk_summary, live = (
        in_order(summary), in_order(model_comparison), in_order(walk_summary), in_order(live))

    deploy = summary.copy()
    deploy["Experiment"] = label(deploy)
    deploy["Interval"] = [f"{lo:.2f} - {hi:.2f}" for lo, hi in zip(
        deploy["Deployment_Test_Balanced_Accuracy_CI95_Low"], deploy["Deployment_Test_Balanced_Accuracy_CI95_High"])]

    walk = walk_summary.copy()
    walk["Experiment"] = label(walk)
    logistic = walk[walk["Model"] == "Logistic Regression"].set_index("Experiment", drop=False)
    naive = walk[walk["Model"] == "Majority Baseline"].set_index("Experiment")
    walk_table = pd.DataFrame(
        {
            "Experiment": logistic["Experiment"].to_numpy(),
            "Days": logistic["Pooled_Predictions"].astype(int).to_numpy(),
            "Naive": naive.reindex(logistic.index)["Pooled_Balanced_Accuracy"].to_numpy(),
            "Logistic": logistic["Pooled_Balanced_Accuracy"].to_numpy(),
            "Interval": [f"{lo:.2f} - {hi:.2f}" for lo, hi in zip(logistic["Pooled_CI95_Low"], logistic["Pooled_CI95_High"])],
            "Above50": logistic["Share_Of_Folds_Above_50"].to_numpy(),
        }
    )

    comparison = model_comparison.copy()
    comparison["Experiment"] = label(comparison)
    comparison["Chosen"] = np.where(comparison["Deployment_Model"].astype(bool), "yes", "")

    forecast = live.copy()
    forecast["Experiment"] = label(forecast)

    findings = key_findings(model_comparison, walk_summary, live)
    findings_html = "".join(f"<li>{html.escape(f)}</li>" for f in findings)
    data_columns = {"Vessel": "Vessel", "Contract": "Contract", "Valid_Observations": "Daily prices",
                    "Start": "First date", "End": "Last date"}
    if set(data_columns).issubset(quality.columns):
        ready = quality[quality["Status"].eq("Ready")] if "Status" in quality else quality
        ready = in_order(ready.copy())
        for column in ("Start", "End"):
            ready[column] = pd.to_datetime(ready[column]).dt.strftime("%Y-%m-%d")
        data_rows = html_table(ready, data_columns, {"Valid_Observations": ",d"})
    else:
        data_rows = ""

    css = """
    :root { --surface:#fcfcfb; --text:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; }
    @media (prefers-color-scheme: dark) { :root { --surface:#1a1a19; --text:#f0efec; --muted:#c3c2b7; --line:#383835; --accent:#3987e5; } figure img { background:#fcfcfb; } }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--surface); color:var(--text); font:15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }
    h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 40px 0 8px; border-top: 1px solid var(--line); padding-top: 24px; }
    p, li { max-width: 78ch; } .muted { color: var(--muted); }
    .findings { border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; }
    .table-wrap { overflow-x: auto; margin: 12px 0; }
    table { border-collapse: collapse; font-size: 13.5px; min-width: 100%; }
    th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { color: var(--muted); font-weight: 600; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
    figure { margin: 16px 0; } figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }
    figure.narrow { max-width: 720px; }
    .badge { display:inline-block; padding: 1px 8px; border-radius: 10px; border: 1px solid var(--line); font-size: 12px; }
    footer { margin-top: 48px; font-size: 13px; }
    @media print { h2 { break-before: auto; } figure { break-inside: avoid; } }
    """

    stale = not bool(live["Signal_Usable"].all()) if not live.empty else False
    sections = f"""
    <h1>Forecasting dry-bulk freight futures (FFA)</h1>
    <p class="muted">Direction of Capesize and Panamax FFA prices 63 and 84 trading days ahead.
    Report generated {datetime.now():%Y-%m-%d %H:%M}.</p>

    <h2>Key findings</h2>
    <ul class="findings">{findings_html}</ul>

    <h2>1. Data and how it is split</h2>
    <p>Each contract is a separate experiment. The data is split in time, never shuffled: models learn on the
    oldest 60%, are chosen on the next 20% and are judged once on the last 20%. A gap of one forecast horizon
    separates the parts, so no training label reaches into a later part.</p>
    {data_rows}
    {embed_image(output_dir, figures['prices'], 'FFA prices with train, validation and test periods')}

    <h2>2. Results on the single held-out test period</h2>
    <p>The model in each row was chosen using validation data only. Balanced accuracy of 0.50 means no skill;
    the interval shows how much the figure could move by chance.</p>
    {html_table(deploy, {"Experiment": "Experiment", "Deployment_Classifier": "Model chosen on validation",
                         "Deployment_Test_Balanced_Accuracy": "Test balanced accuracy", "Interval": "95% interval",
                         "Test_Observations": "Test days"},
                {"Deployment_Test_Balanced_Accuracy": ".2f"}, shade=("Deployment_Test_Balanced_Accuracy",))}
    {embed_image(output_dir, figures['test_accuracy'], 'Test balanced accuracy for every model with intervals')}
    {embed_image(output_dir, figures['validation_vs_test'], 'Validation versus test balanced accuracy', narrow=True)}

    <h2>3. Walk-forward check: does it hold up over time?</h2>
    <p>The whole evaluation is repeated on every consecutive six-month block, retraining each time on data
    available before that block. Pooled balanced accuracy uses all out-of-sample days. The naive baseline
    predicts the direction that dominated the training window, so in trending markets it can beat 0.50.</p>
    {html_table(walk_table, {"Experiment": "Experiment", "Days": "Out-of-sample days", "Naive": "Naive baseline",
                             "Logistic": "Logistic regression", "Interval": "95% interval",
                             "Above50": "Share of blocks above 0.50"},
                {"Days": ",d", "Naive": ".2f", "Logistic": ".2f", "Above50": ".0%"}, shade=("Logistic", "Naive"))}
    {embed_image(output_dir, figures['walk_forward'], 'Walk-forward balanced accuracy per block over time')}

    <h2>4. What the model predicted in the test period</h2>
    <p>Each panel shows the chosen model's probability of a rise on every test day. Ticks along the top and
    bottom mark what actually happened. The grey band is where the strategy would not trade.</p>
    {embed_image(output_dir, figures['test_predictions'], 'Predicted probability of a rise against actual outcomes')}

    <h2>5. Latest forecast</h2>
    <p>{'<span class="badge">Stale data</span> The source files end on ' + html.escape(str(live['As_Of'].max())) + '. These forecasts illustrate the output format and must not be traded.' if stale else 'Forecasts from the most recent data.'}</p>
    {html_table(forecast, {"Experiment": "Contract", "As_Of": "Data as of", "Current_FFA_Price": "Price ($/day)",
                           "Probability_Up": "Probability of a rise", "Predicted_Future_Price": "Predicted price ($/day)",
                           "Signal": "Signal", "Deployment_Classifier": "Model"},
                {"Current_FFA_Price": ",.0f", "Probability_Up": ".0%", "Predicted_Future_Price": ",.0f"})}
    {embed_image(output_dir, figures['latest'], 'Latest probability of a rise per contract', narrow=True)}

    <h2>6. Every model, every experiment</h2>
    {html_table(comparison, {"Experiment": "Experiment", "Model": "Model", "Validation_Balanced_Accuracy": "Validation",
                             "Test_Balanced_Accuracy": "Test", "Test_Balanced_Accuracy_CI95_Low": "CI low",
                             "Test_Balanced_Accuracy_CI95_High": "CI high", "Chosen": "Chosen"},
                {"Validation_Balanced_Accuracy": ".2f", "Test_Balanced_Accuracy": ".2f",
                 "Test_Balanced_Accuracy_CI95_Low": ".2f", "Test_Balanced_Accuracy_CI95_High": ".2f"},
                shade=("Validation_Balanced_Accuracy", "Test_Balanced_Accuracy"))}

    <h2>Limitations</h2>
    <ul>
      <li>Backtests contain only 4-10 trades per test period, so PnL and Sharpe are not evidence and are left out here.</li>
      <li>Overlapping 63/84-day labels are strongly autocorrelated: the effective sample is far smaller than the row count.</li>
      <li>Results vary a lot between six-month blocks; a model with a good average can still have a bad half-year.</li>
    </ul>
    <footer class="muted">Data: Baltic Exchange FFA history (licensed, not distributed). Code:
    github.com/dimositychalasportfolio/ffa-freight-forecasting</footer>
    """
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FFA Forecast Report</title><style>{css}</style></head>
<body><main>{sections}</main></body></html>"""
    path = output_dir / "FFA_Report.html"
    path.write_text(page, encoding="utf-8")
    return path.name


# ---------------------------------------------------------------------------
# Excel workbook
# ---------------------------------------------------------------------------

def build_excel(
    output_dir: Path,
    summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    walk_summary: pd.DataFrame,
    walk_folds: pd.DataFrame,
    live: pd.DataFrame,
    quality: pd.DataFrame,
    results: list["ExperimentResult"],
) -> str:
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    predictions = pd.concat(
        [
            r.test_predictions.assign(Vessel=r.summary["Vessel"], Contract=r.summary["Contract"],
                                      Model=r.summary["Deployment_Classifier"])
            .reset_index(names="Date")
            for r in results
        ],
        ignore_index=True,
    ) if results else pd.DataFrame()

    sheets = {
        "Test results": summary,
        "Model comparison": model_comparison,
        "Walk-forward summary": walk_summary,
        "Walk-forward blocks": walk_folds,
        "Latest forecasts": live,
        "Test predictions": predictions,
        "Data quality": quality,
    }
    path = output_dir / "FFA_Results.xlsx"
    header_fill = PatternFill("solid", fgColor="2A78D6")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = header_fill
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            for index, column in enumerate(frame.columns, start=1):
                letter = get_column_letter(index)
                width = max(len(str(column)), *(len(str(v)) for v in frame[column].head(200))) if len(frame) else len(column)
                sheet.column_dimensions[letter].width = min(max(width + 2, 9), 42)
                if pd.api.types.is_float_dtype(frame[column]):
                    for cell in sheet[letter][1:]:
                        cell.number_format = "0.000"
                if "Balanced_Accuracy" in str(column) and len(frame):
                    sheet.conditional_formatting.add(
                        f"{letter}2:{letter}{len(frame) + 1}",
                        ColorScaleRule(start_type="num", start_value=0.3, start_color="F3B6B5",
                                       mid_type="num", mid_value=0.5, mid_color="FFFFFF",
                                       end_type="num", end_value=0.8, end_color="86B6EF"),
                    )
            sheet.row_dimensions[1].height = 32
    return path.name


# ---------------------------------------------------------------------------
# Entry point used by ffa_panel.main()
# ---------------------------------------------------------------------------

def build_reports(
    results: list["ExperimentResult"],
    summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    walk_forward_folds: pd.DataFrame,
    walk_forward_summary: pd.DataFrame,
    live: pd.DataFrame,
    quality: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Write every chart, the HTML report and the Excel workbook; return the file names."""
    if not results:
        return []
    results = ordered_results(results)
    folder = output_dir / FIGURES_DIR
    figures = {
        "prices": chart_prices_and_splits(results, folder),
        "test_accuracy": chart_test_accuracy(model_comparison, folder),
        "walk_forward": chart_walk_forward(results, walk_forward_folds, folder),
        "validation_vs_test": chart_validation_vs_test(model_comparison, folder),
        "test_predictions": chart_test_predictions(results, folder),
        "latest": chart_latest_forecast(live, folder),
    }
    html_name = build_html(output_dir, figures, summary, model_comparison, walk_forward_summary, live, quality)
    excel_name = build_excel(output_dir, summary, model_comparison, walk_forward_summary, walk_forward_folds,
                             live, quality, results)
    return [html_name, excel_name, *figures.values()]
