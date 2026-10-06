"""Validation of labelled inputs: market data, baseline, initial portfolio, config."""

import unittest

import numpy as np
import pandas as pd

from simple_dynamic_portfolios import (
    InitialPortfolio,
    MarketInputs,
    MarkowitzConfig,
    StaticBaseline,
)

DATES = pd.bdate_range("2020-01-01", periods=6)


def prices(**overrides) -> pd.DataFrame:
    """Two-asset price panel."""
    frame = pd.DataFrame({"x": np.linspace(100, 105, 6), "y": np.linspace(50, 51, 6)}, index=DATES)
    for column, values in overrides.items():
        frame[column] = values
    return frame


CASH = pd.Series(1e-4, index=DATES)


class TestMarketInputs(unittest.TestCase):
    """Dates, labels, and values are checked once, at the boundary."""

    def test_valid_inputs_and_one_column_cash_frame(self) -> None:
        """A one-column DataFrame of cash returns is accepted."""
        market = MarketInputs(prices(), CASH.to_frame("rf"))
        self.assertIsInstance(market.cash_returns, pd.Series)
        self.assertEqual(list(market.assets), ["x", "y"])

    def test_bad_dates_are_rejected(self) -> None:
        """Unsorted, duplicated, or non-datetime indexes fail."""
        cases = {
            "unsorted": prices().iloc[::-1],
            "duplicate": prices().iloc[[0, 0, 1, 2]],
            "not datetime": prices().reset_index(drop=True),
        }
        for name, frame in cases.items():
            with self.subTest(name), self.assertRaises((ValueError, TypeError)):
                MarketInputs(frame, CASH)

    def test_timezone_conventions_must_match(self) -> None:
        """Naive prices with tz-aware cash is ambiguous."""
        with self.assertRaisesRegex(ValueError, "timezone"):
            MarketInputs(prices(), CASH.tz_localize("UTC"))

    def test_bad_values_are_rejected(self) -> None:
        """Prices must be positive and finite; labels unique and not reserved."""
        with self.assertRaises(ValueError):
            MarketInputs(prices(x=[100, 101, -1, 102, 103, 104]), CASH)
        with self.assertRaises(ValueError):
            MarketInputs(prices(x=[100, 101, np.inf, 102, 103, 104]), CASH)
        with self.assertRaises(ValueError):
            MarketInputs(prices().rename(columns={"y": "x"}), CASH)
        with self.assertRaisesRegex(ValueError, "reserved"):
            MarketInputs(prices().rename(columns={"y": "Cash"}), CASH)
        with self.assertRaises(ValueError):
            MarketInputs(prices(), CASH.where(CASH.index != DATES[2], -1.5))

    def test_complete_panel_required_only_inside_the_simulation_interval(self) -> None:
        """Missing history before the start is warm-up; a gap inside is an error."""
        late = prices(y=[np.nan, np.nan, 50, 50.5, 51, 51.5])
        market = MarketInputs(late, CASH)
        data = market.to_backtest_data("M", start=DATES[2])
        self.assertEqual(len(data.timeline), 4)
        with self.assertRaisesRegex(ValueError, "complete price panel"):
            market.to_backtest_data("M")

    def test_cash_is_required_on_every_simulation_date(self) -> None:
        """Cash returns are not silently filled."""
        market = MarketInputs(prices(), CASH.drop(DATES[3]))
        with self.assertRaisesRegex(ValueError, "cash_returns are missing"):
            market.to_backtest_data("M")


class TestStaticBaseline(unittest.TestCase):
    """Strategic weights are validated, never silently normalized."""

    def test_weights_must_sum_to_one_and_be_nonnegative(self) -> None:
        """Malformed baselines fail."""
        for bad in ({"x": 0.6, "y": 0.3}, {"x": 1.2, "y": -0.2}, {"x": np.nan, "y": 1.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                StaticBaseline(pd.Series(bad))

    def test_alignment_fills_omitted_assets_and_rejects_unknown_ones(self) -> None:
        """Omitted assets are zero-baseline; unknown labels (including cash) are errors."""
        assets = pd.Index(["z", "x", "y"])
        np.testing.assert_array_equal(StaticBaseline(pd.Series({"x": 0.7, "y": 0.3})).aligned(
            assets), [0.0, 0.7, 0.3])
        with self.assertRaisesRegex(ValueError, "outside the investable universe"):
            StaticBaseline(pd.Series({"x": 0.7, "Cash": 0.3})).aligned(assets)

    def test_normalized_is_explicit(self) -> None:
        """Holdings-derived baselines go through the named helper."""
        baseline = StaticBaseline.normalized(pd.Series({"x": 30.0, "y": 10.0}))
        self.assertEqual(baseline.weights.to_dict(), {"x": 0.75, "y": 0.25})


class TestInitialPortfolio(unittest.TestCase):
    """Initial state from weights or market values."""

    def test_from_values(self) -> None:
        """Market values including cash normalize to weights."""
        state = InitialPortfolio.from_values(pd.Series({"x": 60.0, "y": 30.0}), cash_value=10.0)
        self.assertAlmostEqual(state.cash_weight, 0.1)
        np.testing.assert_allclose(state.aligned(pd.Index(["y", "x"])), [0.3, 0.6])

    def test_weights_and_cash_must_sum_to_one(self) -> None:
        """An incomplete state is rejected."""
        with self.assertRaises(ValueError):
            InitialPortfolio.from_weights(pd.Series({"x": 0.5}), cash_weight=0.4)


class TestMarkowitzConfig(unittest.TestCase):
    """Units and cost configuration."""

    def test_daily_vol_target_and_shared_spread(self) -> None:
        """The simulator's spread is the optimizer's unless explicitly overridden."""
        config = MarkowitzConfig(annual_vol_target=0.07, bid_ask_spread=1e-3)
        self.assertEqual(config.daily_vol_target, 0.07 / np.sqrt(252))
        self.assertEqual(config.optimizer_spread, 1e-3)
        self.assertEqual(MarkowitzConfig(optimizer_bid_ask_spread=0.0).optimizer_spread, 0.0)

    def test_invalid_settings_are_rejected(self) -> None:
        """Domains are checked at construction."""
        for kwargs in ({"annual_vol_target": 0.0}, {"relative_l1_radius": -1.0},
                       {"rebalance_frequency": "H"}, {"objective_horizon_days": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MarkowitzConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
