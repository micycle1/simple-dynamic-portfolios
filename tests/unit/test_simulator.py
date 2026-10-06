"""Characterization tests for the untaxed simulator's accounting and event order."""

import unittest

import numpy as np
import pandas as pd

from simple_dynamic_portfolios.simulator import BacktestData, rebalance_schedule, run_backtest

SPREAD = 1e-3


class Target:
    """Constructor returning a fixed target and recording what it saw."""

    def __init__(self, weights) -> None:
        self.weights = np.asarray(weights, dtype=float)
        self.seen: list[tuple[pd.Timestamp, np.ndarray, float]] = []

    def __call__(self, ts, curr_weights, universe, ffr):
        self.seen.append((pd.Timestamp(ts), curr_weights.copy(), ffr))
        return self.weights.copy()


def flat_data(n_days: int = 5, n_assets: int = 2, freq: str = "Y", cash: float = 0.0,
              prices: pd.DataFrame | None = None) -> BacktestData:
    """Constant prices (unless given) and constant daily cash returns."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    if prices is None:
        prices = pd.DataFrame(100.0, index=dates, columns=[f"a{j}" for j in range(n_assets)])
    ffr = pd.Series(cash, index=prices.index)
    return BacktestData.from_pandas(prices.index[0], prices.index[-1], prices, ffr, freq)


class TestEventOrder(unittest.TestCase):
    """The legacy sequence of accrual, returns, NAV, rebalance, and costs."""

    def test_first_observation_has_zero_asset_and_cash_return(self) -> None:
        """Day 0 NAV is 1 however the first prices and cash rate look."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"a0": [50.0, 55.0, 60.0]}, index=dates)
        data = flat_data(prices=prices, cash=1e-3)
        self.assertEqual(data.ffrs[0], 0.0)
        np.testing.assert_array_equal(data.returns[0], 0.0)
        result = run_backtest("t", data, Target([0.0]), bid_ask_spread=SPREAD)
        self.assertEqual(result.navs.iloc[0], 1.0)
        self.assertAlmostEqual(result.navs.iloc[1], 1.001)

    def test_costs_reach_the_next_nav_not_the_rebalance_nav(self) -> None:
        """A rebalance's cost is debited from cash after that day's NAV is recorded."""
        data = flat_data(n_assets=1)
        result = run_backtest("t", data, Target([1.0]), bid_ask_spread=SPREAD)
        self.assertEqual(result.navs.iloc[0], 1.0)
        self.assertAlmostEqual(result.navs.iloc[1], 1.0 - 0.5 * SPREAD, places=15)
        self.assertAlmostEqual(result.turnover.iloc[0], 0.5)

    def test_composition_records_pre_cost_targets(self) -> None:
        """Rebalance-day composition is the target, so a fully invested target shows zero cash."""
        result = run_backtest("t", flat_data(n_assets=1), Target([1.0]), bid_ask_spread=SPREAD)
        self.assertEqual(result.composition["a0"].iloc[0], 1.0)
        self.assertEqual(result.composition["Cash"].iloc[0], 0.0)
        # The day after, weights reflect negative cash from the unfunded cost.
        self.assertGreater(result.composition["a0"].iloc[1], 1.0)

    def test_default_start_is_all_cash(self) -> None:
        """Without an initial portfolio the first solve sees zero risky weights."""
        target = Target([0.5, 0.5])
        run_backtest("t", flat_data(), target, bid_ask_spread=SPREAD)
        np.testing.assert_array_equal(target.seen[0][1], 0.0)

    def test_partial_last_period_is_a_rebalance_date(self) -> None:
        """The last observation of a trailing partial period is scheduled."""
        dates = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
        np.testing.assert_array_equal(rebalance_schedule(dates, "M"), [False, True, False, True])

    def test_invalid_constructor_output_fails_clearly(self) -> None:
        """A missing or non-finite target raises instead of corrupting the path."""
        for bad in (None, [np.nan, 0.0], [1.0]):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                run_backtest("t", flat_data(), lambda bad=bad, **_: bad)


class TestInitialPortfolio(unittest.TestCase):
    """An explicit starting state is traded out of like any other rebalance."""

    def test_first_trade_pays_transition_turnover_and_costs(self) -> None:
        """Turnover and cost follow from the initial and target weights."""
        target = Target([0.5, 0.5])
        result = run_backtest("t", flat_data(), target, bid_ask_spread=SPREAD,
                              initial_weights=np.array([0.6, 0.3]))
        np.testing.assert_allclose(target.seen[0][1], [0.6, 0.3])
        self.assertAlmostEqual(result.turnover.iloc[0], 0.5 * (0.1 + 0.2))
        self.assertAlmostEqual(result.navs.iloc[0], 1.0)
        self.assertAlmostEqual(result.navs.iloc[1], 1.0 - 0.5 * SPREAD * 0.3, places=15)

    def test_holding_the_initial_portfolio_costs_nothing(self) -> None:
        """A target equal to the initial state trades nothing."""
        result = run_backtest("t", flat_data(), Target([0.6, 0.3]), bid_ask_spread=SPREAD,
                              initial_weights=np.array([0.6, 0.3]))
        np.testing.assert_allclose(result.turnover, 0.0)
        np.testing.assert_allclose(result.navs, 1.0)

    def test_initial_weights_are_validated(self) -> None:
        """Shape and finiteness are checked."""
        with self.assertRaises(ValueError):
            run_backtest("t", flat_data(), Target([0.5, 0.5]), initial_weights=np.ones(3))


class TestForcedLiquidation(unittest.TestCase):
    """Legacy handling of assets that become unavailable (only via ``from_pandas``)."""

    def test_cost_without_rebalance_is_not_persisted(self) -> None:
        """Known accounting characteristic: the liquidation cost reverses the next day.

        The cost lowers that day's recorded NAV but is never debited from cash.
        Preserved for reproduction; labelled inputs forbid missing prices, so
        the public pipeline cannot reach this path.
        """
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame(100.0, index=dates, columns=["a0", "a1"])
        prices.iloc[2:, 1] = np.nan
        data = flat_data(prices=prices)
        result = run_backtest("t", data, Target([0.5, 0.5]), bid_ask_spread=SPREAD)
        navs = result.navs.to_numpy()
        liquidated = 0.5  # dollars held in a1, bought on day 0
        self.assertAlmostEqual(navs[2], navs[1] - SPREAD * liquidated, places=15)
        self.assertAlmostEqual(navs[3], navs[1], places=15)
        self.assertAlmostEqual(result.turnover.iloc[2], liquidated / navs[2])


if __name__ == "__main__":
    unittest.main()
