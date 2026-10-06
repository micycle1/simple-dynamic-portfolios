"""Tests for portfolio constructors."""

import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from sbg_portfolios.optimizer import (
    AnchoredVolControlPortfolioConstructor,
    SolverError,
    solve_for_weights,
)


class TestAnchoredVolControlPortfolioConstructor(unittest.TestCase):
    """Tests for the anchored Markowitz objective terms."""

    def make_constructor(
        self, alphas: np.ndarray, *, spread: float = 0.0, cash_days: int | None = None
    ) -> tuple[AnchoredVolControlPortfolioConstructor, pd.Timestamp]:
        """Build a loose-risk, two-asset problem for objective tests."""
        ts = pd.Timestamp("2020-01-31")
        risk_model = {
            "Fs": {ts: np.zeros((2, 1))},
            "D_halves": {ts: np.full(2, 1e-3)},
        }
        constructor = AnchoredVolControlPortfolioConstructor(
            ts_lookup={ts: 0},
            alphas=alphas[None, :],
            risk_model=risk_model,
            vol_target=1.0,
            anchor=np.array([0.5, 0.5]),
            bid_ask_spread=spread,
            cash_rate_horizon_days=cash_days,
        )
        return constructor, ts

    def solve(
        self,
        constructor: AnchoredVolControlPortfolioConstructor,
        ts: pd.Timestamp,
        current: np.ndarray,
        daily_ffr: float = 0.0,
    ) -> np.ndarray:
        """Solve a fully available synthetic portfolio."""
        return constructor(ts, current, np.ones(2, dtype=bool), daily_ffr)

    def test_monthly_cash_return_can_dominate_risky_assets(self) -> None:
        """Cash is selected when its compounded horizon return exceeds alpha."""
        constructor, ts = self.make_constructor(np.array([0.001, 0.001]), cash_days=21)
        weights = self.solve(constructor, ts, np.zeros(2), daily_ffr=1e-4)
        np.testing.assert_allclose(weights, 0.0, atol=1e-6)

    def test_half_spread_penalty_uses_current_weights(self) -> None:
        """A forecast edge smaller than the half-spread does not justify trading."""
        current = np.array([0.5, 0.5])
        constructor, ts = self.make_constructor(
            np.array([0.0011, 0.0010]), spread=5e-4, cash_days=21
        )
        weights = self.solve(constructor, ts, current)
        np.testing.assert_allclose(weights, current, atol=1e-6)

    def test_invalid_objective_parameters_are_rejected(self) -> None:
        """Financial objective parameters must have valid domains."""
        with self.assertRaises(ValueError):
            self.make_constructor(np.ones(2), spread=-1e-4)
        with self.assertRaises(ValueError):
            self.make_constructor(np.ones(2), cash_days=0)


class TestAnchorRadius(unittest.TestCase):
    """The trust region ||w - g a||_1 <= rho g around the baseline."""

    TS = pd.Timestamp("2020-01-31")

    def solve(self, alphas, anchor, radius=None) -> np.ndarray:
        """Loose-risk problem; ``radius=None`` uses the default."""
        n = len(anchor)
        kwargs = {} if radius is None else {"relative_l1_radius": radius}
        constructor = AnchoredVolControlPortfolioConstructor(
            ts_lookup={self.TS: 0},
            alphas=np.asarray(alphas, dtype=float)[None, :],
            risk_model={"Fs": {self.TS: np.zeros((n, 1))}, "D_halves": {self.TS: np.full(n, 1e-3)}},
            vol_target=1.0,
            anchor=np.asarray(anchor, dtype=float),
            **kwargs,
        )
        return constructor(self.TS, np.zeros(n), np.ones(n, dtype=bool), 0.0)

    def test_zero_radius_holds_baseline_proportions(self) -> None:
        """With rho = 0, risky proportions equal the baseline whatever alpha says."""
        weights = self.solve([0.05, 0.01, 0.02], [0.5, 0.3, 0.2], radius=0.0)
        np.testing.assert_allclose(weights / weights.sum(), [0.5, 0.3, 0.2], atol=1e-6)

    def test_default_radius_is_the_paper_constraint(self) -> None:
        """The default radius is one, and the constraint binds at it."""
        anchor = np.array([0.5, 0.3, 0.2])
        default = self.solve([0.05, 0.01, 0.02], anchor)
        np.testing.assert_allclose(default, self.solve([0.05, 0.01, 0.02], anchor, 1.0), atol=1e-7)
        gross = default.sum()
        self.assertAlmostEqual(np.abs(default - gross * anchor).sum(), gross, places=5)

    def test_zero_baseline_asset_can_receive_tactical_weight(self) -> None:
        """An investable asset outside the baseline is reachable only when rho > 0."""
        anchor = [0.5, 0.5, 0.0]
        self.assertGreater(self.solve([0.0, 0.0, 0.05], anchor, 0.5)[2], 0.1)
        self.assertAlmostEqual(self.solve([0.0, 0.0, 0.05], anchor, 0.0)[2], 0.0, places=6)

    def test_negative_radius_is_rejected(self) -> None:
        """The radius must be nonnegative."""
        with self.assertRaises(ValueError):
            self.solve([0.0, 0.0], [0.5, 0.5], radius=-0.1)


class TestSolverFailure(unittest.TestCase):
    """Solver problems raise instead of returning None downstream."""

    def test_infeasible_problem_raises(self) -> None:
        """A non-optimal status is an error, with no fallback portfolio."""
        x = cp.Variable(2)
        problem = cp.Problem(cp.Maximize(cp.sum(x)), [x >= 1.0, cp.sum(x) <= 0.0])
        with self.assertRaisesRegex(SolverError, "infeasible"):
            solve_for_weights(problem, x, pd.Timestamp("2020-01-31"))


if __name__ == "__main__":
    unittest.main()
