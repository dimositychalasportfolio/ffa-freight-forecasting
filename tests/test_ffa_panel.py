"""Checks for the parts of the pipeline where a silent mistake would leak the future
or misstate a result. Run with: python -m pytest"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ffa_panel  # noqa: E402


def synthetic_prices(n: int = 1_200, seed: int = 0) -> pd.Series:
    """A random-walk price series on business days, enough for every split."""
    rng = np.random.default_rng(seed)
    values = 10_000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.Series(values, index=pd.bdate_range("2015-01-01", periods=n), name="Price")


def test_features_do_not_use_future_prices():
    """Changing prices after day t must not change any feature on day t."""
    prices = synthetic_prices()
    cutoff = 800
    changed = prices.copy()
    changed.iloc[cutoff + 1:] *= 3

    original, _ = ffa_panel.make_feature_frame(prices, 63)
    altered, _ = ffa_panel.make_feature_frame(changed, 63)
    day = prices.index[cutoff]
    pd.testing.assert_series_equal(original.loc[day, ffa_panel.FEATURES], altered.loc[day, ffa_panel.FEATURES])


def test_price_level_is_not_a_feature():
    assert "Price" not in ffa_panel.FEATURES


def test_target_is_one_when_price_rises_over_the_horizon():
    frame, _ = ffa_panel.make_feature_frame(synthetic_prices(), 63)
    expected = (frame["Future_Price"] > frame["Price"]).astype(int)
    pd.testing.assert_series_equal(frame["Target"], expected, check_names=False)


@pytest.mark.parametrize("horizon", [63, 84])
def test_chronological_split_purges_labels_before_each_boundary(horizon):
    frame, _ = ffa_panel.make_feature_frame(synthetic_prices(), horizon)
    train, validation, test = ffa_panel.chronological_split(frame, horizon)
    position = {date: i for i, date in enumerate(frame.index)}
    # A row's label uses the price `horizon` rows later; it must land before the next block.
    assert position[train.index[-1]] + horizon < position[validation.index[0]]
    assert position[validation.index[-1]] + horizon < position[test.index[0]]


@pytest.mark.parametrize("horizon", [63, 84])
def test_walk_forward_folds_are_purged_and_move_forward(horizon):
    folds = ffa_panel.walk_forward_folds(2_000, horizon)
    assert len(folds) > 1
    for train_rows, test_rows in folds:
        assert train_rows.start == 0
        assert len(train_rows) >= ffa_panel.WALK_FORWARD_MIN_TRAIN_DAYS
        assert test_rows.start - train_rows.stop == horizon
        assert len(test_rows) == ffa_panel.WALK_FORWARD_TEST_DAYS
    test_starts = [test_rows.start for _, test_rows in folds]
    assert test_starts == sorted(test_starts)
    # Test blocks follow each other without overlap.
    for (_, previous), (_, current) in zip(folds, folds[1:]):
        assert current.start == previous.stop


def test_walk_forward_returns_no_folds_when_data_is_too_short():
    assert ffa_panel.walk_forward_folds(300, 63) == []


def test_sharpe_is_annualised_by_trade_frequency():
    pnl = pd.Series([1.0, 2.0, -1.0, 3.0])
    per_trade = pnl.mean() / pnl.std(ddof=1)
    assert ffa_panel.sharpe_ratio(pnl, 63) == pytest.approx(per_trade * np.sqrt(4))
    assert ffa_panel.sharpe_ratio(pnl, 84) == pytest.approx(per_trade * np.sqrt(3))


def test_sharpe_is_nan_without_variation():
    assert np.isnan(ffa_panel.sharpe_ratio(pd.Series([5.0, 5.0, 5.0]), 63))


def test_bootstrap_interval_for_perfect_predictions_is_one():
    truth = pd.Series([0, 1] * 100)
    low, high = ffa_panel.block_bootstrap_ci(truth, truth.to_numpy(), block_length=10, samples=200)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    truth = pd.Series(rng.integers(0, 2, 400))
    prediction = np.where(rng.random(400) < 0.7, truth, 1 - truth)
    point = ffa_panel.balanced_accuracy_score(truth, prediction)
    low, high = ffa_panel.block_bootstrap_ci(truth, prediction, block_length=20, samples=500)
    assert low <= point <= high


def test_backtest_trades_once_per_horizon_and_charges_costs():
    frame, _ = ffa_panel.make_feature_frame(synthetic_prices(), 63)
    test = frame.iloc[-300:]
    always_up = np.full(len(test), 0.9)
    backtest = ffa_panel.non_overlapping_backtest(test, always_up, threshold=0.55, trading_days=63)
    assert len(backtest) == int(np.ceil(len(test) / 63))
    assert (backtest["Signal"] == 1).all()
    expected_cost = 2 * ffa_panel.TRANSACTION_COST_POINTS * backtest["Size"]
    pd.testing.assert_series_equal(backtest["Cost"], expected_cost, check_names=False)
