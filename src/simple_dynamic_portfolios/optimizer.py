"""Portfolio constructors: fixed-weight benchmarks and factor-risk optimizers.

Constructors use the positional calling convention of ``run_backtest``: arrays
aligned to one asset order, a ``ts_lookup`` from decision date to row, and a
risk model ``{"Fs": {ts: F}, "D_halves": {ts: d}}`` in daily units, with
covariance ``F @ F.T + diag(d**2)``.
"""

from collections.abc import Mapping
from typing import Protocol

import cvxpy as cp
import numpy as np
import pandas as pd

# ruff: noqa: ARG002


EPSILON = 1e-6


class SolverError(RuntimeError):
    """The optimizer did not return a usable solution."""


def solve_for_weights(
    problem: cp.Problem,
    weights: cp.Variable,
    ts: pd.Timestamp,
    solver: str = cp.CLARABEL,
    solver_options: Mapping | None = None,
    accept_inaccurate: bool = False,
) -> np.ndarray:
    """Solve ``problem`` and return ``weights``, failing clearly instead of returning None."""
    try:
        problem.solve(solver=solver, verbose=False, **dict(solver_options or {}))
    except cp.error.SolverError as exc:
        raise SolverError(f"{solver} failed on {pd.Timestamp(ts)}: {exc}") from exc
    accepted = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} if accept_inaccurate else {cp.OPTIMAL}
    if problem.status not in accepted or weights.value is None:
        raise SolverError(
            f"{solver} returned status {problem.status!r} on {pd.Timestamp(ts)}; "
            "no fallback portfolio is used."
        )
    return weights.value


class PortfolioConstructor(Protocol):
    """Portfolio constructor."""

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        raise NotImplementedError


class FixedWeightPortfolioConstructor:
    """Fixed weight portfolio constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        fixed_weights: np.ndarray,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the fixed weight portfolio constructor."""
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(fixed_weights), 1))

        self.leverage = leverage
        self.ts_lookup = ts_lookup
        self.fixed_weights = fixed_weights
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        weights = self.fixed_weights[idx]
        universe = universe if self.universe is None else universe * self.universe[idx]

        unnorm_weights = weights * universe
        if np.abs(unnorm_weights).sum() < EPSILON:
            return np.zeros_like(weights)

        return self.leverage * unnorm_weights / np.abs(unnorm_weights).sum()


class FixedWeightVolControlPortfolioConstructor:
    """Fixed weight portfolio with volatility control constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        vols: np.ndarray,
        fixed_weights: np.ndarray,
        vol_target: float,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the fixed weight portfolio with volatility control constructor."""
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if vol_target <= 0.0:
            raise ValueError("Volatility target must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(fixed_weights), 1))

        self.leverage = leverage
        self.vol_target = vol_target
        self.ts_lookup = ts_lookup
        self.vols = vols
        self.fixed_weights = fixed_weights
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        fixed_weights = self.fixed_weights[idx]

        universe = universe if self.universe is None else universe * self.universe[idx]

        if np.abs(fixed_weights[universe]).sum() < EPSILON:
            return np.zeros_like(fixed_weights)

        weights = (fixed_weights * universe) / np.abs(fixed_weights[universe]).sum()
        vol = self.vols[idx]

        if vol < EPSILON:
            return np.zeros_like(fixed_weights)

        scaling = np.clip(self.vol_target / vol, 0.0, self.leverage)
        return weights * scaling


class FixedWeightMatrixVolControlPortfolioConstructor:
    """Fixed weight portfolio with volatility control constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        fixed_weights: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the fixed weight portfolio with volatility control constructor."""
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if vol_target <= 0.0:
            raise ValueError("Volatility target must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(fixed_weights), 1))

        self.leverage = leverage
        self.vol_target = vol_target
        self.ts_lookup = ts_lookup
        self.fixed_weights = fixed_weights
        self.risk_model = risk_model
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        fixed_weights = self.fixed_weights[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]

        if np.abs(fixed_weights[universe]).sum() < EPSILON:
            return np.zeros_like(fixed_weights)

        weights = (fixed_weights * universe) / np.abs(fixed_weights[universe]).sum()
        f = np.concatenate([F.T @ weights, D_half * weights])
        total_vol = np.linalg.norm(f)

        if total_vol < EPSILON:
            return np.zeros_like(fixed_weights)

        scaling = np.clip(self.vol_target / total_vol, 0.0, self.leverage)
        return weights * scaling


class VolControlPortfolioConstructor:
    """Volatility control portfolio constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        alphas: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
        solver: str = cp.CLARABEL,
        solver_options: Mapping | None = None,
        accept_inaccurate: bool = False,
    ) -> None:
        """Initialize the volatility control portfolio constructor."""
        if vol_target <= 0.0:
            raise ValueError("Volatility target must be positive.")
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(alphas), 1))

        self.solver = solver
        self.solver_options = dict(solver_options or {})
        self.accept_inaccurate = accept_inaccurate
        self.leverage = leverage
        self.vol_target = vol_target
        self.ts_lookup = ts_lookup
        self.risk_model = risk_model
        self._alphas = alphas
        self.n = None
        self.k = None
        self.Sig_half = None
        self.alpha_param = None
        self.mask = None
        self.weights = None
        self.problem = None
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        alphas = self._alphas[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]
        n, k = F.shape

        if (n != self.n) or (k != self.k):
            self.n = n
            self.k = k
            self.Sig_half = cp.Parameter(shape=(n + k, n))
            self.alpha_param = cp.Parameter(shape=(n,))
            self.mask = cp.Parameter(shape=(n,))
            self.weights = cp.Variable(n)
            self.objective = cp.Maximize(cp.scalar_product(self.alpha_param, self.weights))
            self.factor_risk = cp.norm2(self.Sig_half @ self.weights)
            self.constraints = [
                cp.sum(self.weights) <= self.leverage,
                self.weights >= 0.0,
                self.factor_risk <= self.vol_target,
                cp.multiply(self.weights, self.mask) == 0.0,
            ]
            self.problem = cp.Problem(self.objective, self.constraints)

        self.Sig_half.value = np.vstack([F.T, np.diag(D_half)])
        self.alpha_param.value = alphas
        self.mask.value = (~universe).astype(float)
        return self._solve(ts)

    def _solve(self, ts: pd.Timestamp) -> np.ndarray:
        return solve_for_weights(
            self.problem,
            self.weights,
            ts,
            solver=self.solver,
            solver_options=self.solver_options,
            accept_inaccurate=self.accept_inaccurate,
        )


class AnchoredVolControlPortfolioConstructor(VolControlPortfolioConstructor):
    r"""Anchored volatility control with optional cash and trading-cost terms.

    With risky weights ``w``, exposure ``g = 1^T w``, anchor ``a``, current
    weights ``w0``, and horizon cash return ``r``, it solves::

        maximize    alpha^T w + r (1 - g) - (s / 2) ||w - w0||_1
        subject to  w >= 0,  g <= leverage,  ||[F^T; diag(d)] w||_2 <= vol_target,
                    w_i = 0 for unavailable assets,
                    ||w - g a||_1 <= relative_l1_radius * g.

    ``s`` is the full ``bid_ask_spread``; ``r`` compounds the decision date's
    daily cash return over ``cash_rate_horizon_days`` (zero when ``None``).
    ``alphas`` must be expected simple returns over that same horizon.
    """

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        alphas: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        anchor: np.ndarray,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
        bid_ask_spread: float = 0.0,
        cash_rate_horizon_days: int | None = None,
        relative_l1_radius: float = 1.0,
        solver: str = cp.CLARABEL,
        solver_options: Mapping | None = None,
        accept_inaccurate: bool = False,
    ) -> None:
        """Initialize the anchored volatility control portfolio constructor."""
        super().__init__(
            ts_lookup=ts_lookup,
            alphas=alphas,
            risk_model=risk_model,
            vol_target=vol_target,
            universe=universe,
            leverage=leverage,
            solver=solver,
            solver_options=solver_options,
            accept_inaccurate=accept_inaccurate,
        )
        if bid_ask_spread < 0.0:
            raise ValueError("Bid-ask spread must be nonnegative.")
        if cash_rate_horizon_days is not None and cash_rate_horizon_days <= 0:
            raise ValueError("Cash-rate horizon must be positive.")
        if not relative_l1_radius >= 0.0:
            raise ValueError("Relative L1 radius must be nonnegative.")
        self.relative_l1_radius = relative_l1_radius
        self.anchor = anchor
        self.bid_ask_spread = bid_ask_spread
        self.cash_rate_horizon_days = cash_rate_horizon_days
        self.curr_weights_param = None
        self.cash_return_param = None

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        alphas = self._alphas[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]
        n, k = F.shape

        if (n != self.n) or (k != self.k):
            self.n = n
            self.k = k
            self.Sig_half = cp.Parameter(shape=(n + k, n))
            self.alpha_param = cp.Parameter(shape=(n,))
            self.mask = cp.Parameter(shape=(n,))
            self.curr_weights_param = cp.Parameter(shape=(n,))
            self.cash_return_param = cp.Parameter()
            self.weights = cp.Variable(n)
            gross = cp.sum(self.weights)
            cash = 1.0 - gross
            expected_net_return = cp.scalar_product(self.alpha_param, self.weights)
            expected_net_return += self.cash_return_param * cash
            expected_net_return -= (
                0.5 * self.bid_ask_spread * cp.norm1(self.weights - self.curr_weights_param)
            )
            self.objective = cp.Maximize(expected_net_return)
            self.factor_risk = cp.norm2(self.Sig_half @ self.weights)
            self.constraints = [
                gross <= self.leverage,
                self.weights >= 0.0,
                self.factor_risk <= self.vol_target,
                cp.multiply(self.weights, self.mask) == 0.0,
                cp.norm1(self.weights - self.anchor * gross) <= self.relative_l1_radius * gross,
            ]
            self.problem = cp.Problem(self.objective, self.constraints)

        self.Sig_half.value = np.vstack([F.T, np.diag(D_half)])
        self.alpha_param.value = alphas
        self.mask.value = (~universe).astype(float)
        self.curr_weights_param.value = curr_weights
        self.cash_return_param.value = (
            0.0
            if self.cash_rate_horizon_days is None
            else np.expm1(self.cash_rate_horizon_days * np.log1p(ffr))
        )
        return self._solve(ts)
