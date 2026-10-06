"""Metric conventions."""

import unittest

import numpy as np
import pandas as pd

from simple_dynamic_portfolios import BacktestResults, arithmetic_sharpe, compute_metrics
from simple_dynamic_portfolios import geometric_excess_sharpe


def results(daily_return: float = 4e-4, n_days: int = 253, seed: int = 1) -> BacktestResults:
    """A NAV path with noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    returns = np.r_[0.0, daily_return + rng.normal(0, 0.005, n_days - 1)]
    navs = pd.Series(np.cumprod(1 + returns), index=dates)
    return BacktestResults(navs, pd.DataFrame(index=dates), pd.Series(0.001, index=dates),
                           pd.Series(dtype=object))


class TestMetrics(unittest.TestCase):
    """Legacy formulas, optional references, and explicit Sharpe conventions."""

    def test_references_are_optional(self) -> None:
        """Rows needing an omitted reference are left out."""
        stats = compute_metrics(results())
        self.assertNotIn("Sharpe Ratio (FFR)", stats)
        self.assertNotIn("Return - CPI", stats)
        self.assertAlmostEqual(stats["Turnover"], 0.001 * 252)

    def test_cash_adjusted_sharpe_is_geometric(self) -> None:
        """The paper's Sharpe is compounded excess CAGR over volatility, not arithmetic."""
        res = results()
        cash = pd.Series(1e-4, index=res.navs.index)
        stats = compute_metrics(res, cash, cash_label="rf")
        n = len(res.navs) - 1
        growth = res.navs.iloc[-1] / res.navs.iloc[0] / (1 + 1e-4) ** n
        vol = res.navs.pct_change().std() * np.sqrt(252)
        self.assertAlmostEqual(stats["Sharpe Ratio (rf)"], (growth ** (252 / n) - 1) / vol)
        self.assertAlmostEqual(stats["Sharpe Ratio (rf)"],
                               geometric_excess_sharpe(res.navs, cash))
        self.assertNotAlmostEqual(stats["Sharpe Ratio (rf)"], arithmetic_sharpe(res.navs, cash),
                                  places=3)


if __name__ == "__main__":
    unittest.main()
