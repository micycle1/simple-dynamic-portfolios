"""End-to-end runs of the library on prepared inputs outside the paper universe."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import simple_portfolio as sp

ROOT = Path(__file__).resolve().parents[2]


def load_example():
    """Import examples/second_universe.py as a module."""
    spec = importlib.util.spec_from_file_location("second_universe",
                                                  ROOT / "examples" / "second_universe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXAMPLE = load_example()


class TestSecondUniverse(unittest.TestCase):
    """Forecasting, risk, allocation, simulation, and evaluation on a new universe."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the example once."""
        cls.results = EXAMPLE.run_example(n_days=1500)

    def test_every_strategy_runs_and_evaluates(self) -> None:
        """Finite NAVs and metrics for every strategy."""
        for name, result in self.results.items():
            with self.subTest(name):
                self.assertTrue(np.isfinite(result.backtest.navs).all())
                self.assertTrue(np.isfinite(result.evaluate().to_numpy(dtype=float)).all())

    def test_optimizer_constraints_hold_at_every_decision(self) -> None:
        """Long-only, exposure cap, anchor trust region, and ex-ante volatility limit."""
        baseline = np.array([0.40, 0.10, 0.35, 0.15, 0.0])
        config_vol = 0.08 / np.sqrt(252)
        for name in ("ridge + rolling risk", "ewma + prepared covariance"):
            result = self.results[name]
            for date, w in result.decisions.iterrows():
                w = w.to_numpy()
                gross = w.sum()
                with self.subTest(name=name, date=date):
                    self.assertGreaterEqual(w.min(), -1e-7)
                    self.assertLessEqual(gross, 1.0 + 1e-7)
                    self.assertLessEqual(np.abs(w - gross * baseline).sum(), 0.5 * gross + 1e-6)
                    vol = np.sqrt(w @ result.risk.covariance(date).to_numpy() @ w)
                    self.assertLessEqual(vol, config_vol * (1 + 1e-6))
            self.assertTrue((result.diagnostics["status"] == "optimal").all())

    def test_zero_baseline_asset_is_used_tactically(self) -> None:
        """The investable asset outside the baseline receives weight via the trust region."""
        self.assertGreater(self.results["ridge + rolling risk"].decisions["commodity"].max(), 0.0)

    def test_initial_portfolio_is_the_first_current_state(self) -> None:
        """The first rebalance trades out of the supplied holdings, paying for it."""
        result = self.results["ridge + rolling risk"]
        first = result.decisions.iloc[0].to_numpy()
        initial = np.array([0.6, 0.0, 0.3, 0.0, 0.0])
        self.assertAlmostEqual(result.backtest.turnover.iloc[0],
                               0.5 * np.abs(first - initial).sum())


class TestAlignment(unittest.TestCase):
    """Results depend on labels, not column positions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Inputs for a short run."""
        cls.market, cls.features = EXAMPLE.synthetic_inputs(n_days=800)
        cls.start = cls.market.total_return_prices.index[300]
        cls.baseline = sp.StaticBaseline(pd.Series({"equity_us": 0.5, "treasury": 0.5}))
        prices = cls.market.total_return_prices
        cls.alpha = sp.EwmaAlpha(halflife=63).forecast(prices)
        cls.risk = sp.RollingSampleCovariance(window=63).estimate(prices)

    def run_with(self, market, alpha=None, risk=None, **kwargs) -> sp.StrategyResult:
        """run_markowitz with the shared inputs."""
        return sp.run_markowitz(market, self.baseline, alpha or self.alpha, risk or self.risk,
                                start=self.start, **kwargs)

    def test_permuting_asset_columns_permutes_nothing_else(self) -> None:
        """Every input is aligned by label to the market's order."""
        base = self.run_with(self.market)
        order = ["credit", "commodity", "treasury", "equity_em", "equity_us"]
        shuffled = sp.MarketInputs(self.market.total_return_prices[order],
                                   self.market.cash_returns)
        other = self.run_with(shuffled)
        np.testing.assert_allclose(other.backtest.navs, base.backtest.navs, rtol=1e-7)
        pd.testing.assert_frame_equal(other.decisions[base.decisions.columns], base.decisions,
                                      atol=1e-6, check_exact=False)

    def test_covariance_input_matches_its_factor_form(self) -> None:
        """A prepared covariance equal to the factor model's gives the same allocations.

        The factorizations differ (eigen vs Cholesky), so agreement is up to
        solver tolerance; with a nearly flat objective, weights move ~1e-5
        along optimal faces and that propagates through the path.
        """
        decision_dates = self.run_with(self.market).decisions.index
        covariances = pd.concat({d: self.risk.covariance(d) for d in decision_dates},
                                names=["date", "asset"])
        via_cov = self.run_with(self.market, risk=sp.CovarianceHistory(covariances))
        via_factors = self.run_with(self.market)
        pd.testing.assert_frame_equal(via_cov.decisions, via_factors.decisions, atol=2e-4,
                                      check_exact=False)
        np.testing.assert_allclose(via_cov.backtest.navs, via_factors.backtest.navs, rtol=1e-4)

    def test_missing_model_inputs_fail_before_simulating(self) -> None:
        """Decision dates without forecasts or risk, or unknown baseline assets, are errors."""
        decision = self.run_with(self.market).decisions.index[3]
        with self.assertRaisesRegex(ValueError, "Alpha forecasts"):
            self.run_with(self.market, alpha=sp.AlphaForecasts(
                self.alpha.values.drop(decision), horizon_days=1))
        risk = self.risk
        kept = risk.dates != decision
        thin = sp.FactorRiskHistory(risk.loadings.loc[risk.dates[kept]],
                                    risk.residual_std.loc[kept])
        with self.assertRaisesRegex(ValueError, "Risk history is missing"):
            self.run_with(self.market, risk=thin)
        with self.assertRaisesRegex(ValueError, "outside the investable universe"):
            sp.run_markowitz(self.market, sp.StaticBaseline(pd.Series({"gold": 1.0})),
                             self.alpha, self.risk, start=self.start)

    def test_default_initial_state_is_all_cash(self) -> None:
        """``initial_portfolio=None`` equals an explicit all-cash state."""
        cash_only = sp.InitialPortfolio.from_weights(pd.Series(dtype=float), cash_weight=1.0)
        a = self.run_with(self.market)
        b = self.run_with(self.market, initial_portfolio=cash_only)
        pd.testing.assert_series_equal(a.backtest.navs, b.backtest.navs)


class TestPackageBoundary(unittest.TestCase):
    """The core imports no vendor, plotting, or paper code."""

    def test_core_import_is_self_contained(self) -> None:
        """Importing the package pulls in none of the application dependencies."""
        code = (
            "import sys, simple_portfolio, simple_portfolio.pipeline\n"
            "bad = [m for m in ('yfinance', 'fredapi', 'matplotlib', 'experiments', 'tqdm')"
            " if m in sys.modules]\n"
            "assert not bad, bad\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT / "src")

    def test_no_tax_or_ticker_logic_in_core(self) -> None:
        """No tax bookkeeping, paper tickers, or paper paths in the library source."""
        source = "\n".join(p.read_text() for p in (ROOT / "src").rglob("*.py")).lower()
        for needle in ("tax_", "long_term", "short_term", "wash", "_lot", "income_returns",
                       "dividend", '"spy"', '"agg"', '"gld"', "fred", "data/raw",
                       "data/processed"):
            self.assertNotIn(needle, source)


if __name__ == "__main__":
    unittest.main()
