"""Risk histories in daily units, their validation, and the paper's estimator.

Every risk history resolves to a factor form ``Sigma_t = F_t F_t^T + diag(d_t^2)``,
which is what the optimizer consumes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .data import check_dates, check_known, check_labels


def _check_panel(frame: pd.DataFrame, what: str) -> tuple[pd.DatetimeIndex, pd.Index]:
    """Validate a DataFrame indexed by ``(date, asset)``; return its dates and assets."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.MultiIndex):
        raise TypeError(f"{what} must be a DataFrame indexed by (date, asset).")
    if frame.index.nlevels != 2:
        raise ValueError(f"{what} index must have exactly two levels: (date, asset).")
    if not frame.index.is_unique:
        raise ValueError(f"{what} has duplicate (date, asset) rows.")
    dates = check_dates(frame.index.get_level_values(0).unique(), f"{what} dates")
    assets = check_labels(frame.index.get_level_values(1).unique(), f"{what} assets")
    if len(frame) != len(dates) * len(assets):
        raise ValueError(f"{what} must list the same assets on every date.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{what} must be finite.")
    return dates, assets


def _require_dates(available: pd.DatetimeIndex, dates: pd.DatetimeIndex, what: str) -> None:
    missing = dates.difference(available)
    if len(missing):
        raise ValueError(
            f"{what} is missing {len(missing)} decision dates, first {missing[0].date()}."
        )


def _require_assets(available: pd.Index, assets: pd.Index, what: str) -> None:
    check_known(available, assets, what)
    missing = [asset for asset in assets if asset not in available]
    if missing:
        raise ValueError(f"{what} is missing assets: {missing}.")


@dataclass(frozen=True)
class FactorRiskHistory:
    """Daily factor risk: loadings by ``(date, asset)`` x factor, residual std by date x asset."""

    loadings: pd.DataFrame
    residual_std: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate structure, finiteness, and matching labels."""
        dates, assets = _check_panel(self.loadings, "Factor loadings")
        residual = self.residual_std
        if not isinstance(residual, pd.DataFrame):
            raise TypeError("residual_std must be a DataFrame (dates x assets).")
        check_dates(residual.index, "Residual std")
        check_labels(residual.columns, "Residual std columns")
        if not residual.index.equals(dates):
            raise ValueError("Loadings and residual std must cover the same dates.")
        if set(residual.columns) != set(assets):
            raise ValueError("Loadings and residual std must cover the same assets.")
        values = residual.to_numpy(dtype=float)
        if not (np.isfinite(values).all() and (values >= 0.0).all()):
            raise ValueError("Residual std must be finite and nonnegative.")

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Dates with a risk estimate."""
        return self.residual_std.index

    def covariance(self, date) -> pd.DataFrame:
        """Reconstructed daily covariance on ``date``."""
        F = self.loadings.loc[pd.Timestamp(date)]
        d = self.residual_std.loc[pd.Timestamp(date), F.index]
        return F @ F.T + np.diag(d.to_numpy() ** 2)

    def to_constructor_dict(self, dates: pd.DatetimeIndex, assets: pd.Index,
                            keys: Sequence | None = None) -> dict[str, dict]:
        """``{"Fs": {key: F}, "D_halves": {key: d}}`` on ``dates`` in ``assets`` order."""
        _require_dates(self.dates, dates, "Risk history")
        _require_assets(self.residual_std.columns, assets, "Risk history")
        keys = list(dates) if keys is None else list(keys)
        fs, d_halves = {}, {}
        for key, date in zip(keys, dates, strict=True):
            fs[key] = self.loadings.loc[date].reindex(assets).to_numpy(dtype=float)
            d_halves[key] = self.residual_std.loc[date, assets].to_numpy(dtype=float)
        return {"Fs": fs, "D_halves": d_halves}

    @classmethod
    def from_arrays(cls, dates: pd.DatetimeIndex, assets: pd.Index, loadings: np.ndarray,
                    residual_std: np.ndarray) -> "FactorRiskHistory":
        """From stacked arrays: ``loadings`` (dates, assets, factors), ``residual_std`` (dates, assets)."""
        n_dates, n_assets, n_factors = loadings.shape
        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        frame = pd.DataFrame(loadings.reshape(n_dates * n_assets, n_factors), index=index,
                             columns=range(n_factors))
        residual = pd.DataFrame(residual_std, index=dates, columns=assets)
        return cls(frame, residual)

    @classmethod
    def from_legacy_dict(cls, risk_model: dict) -> "FactorRiskHistory":
        """From the paper's pickled ``{"Fs": {ts: F}, "D_halves": {ts: d}}`` of pandas objects."""
        loadings = pd.concat({pd.Timestamp(ts): pd.DataFrame(F)
                              for ts, F in risk_model["Fs"].items()}, names=["date", "asset"])
        residual = pd.DataFrame({pd.Timestamp(ts): pd.Series(d)
                                 for ts, d in risk_model["D_halves"].items()}).T
        return cls(loadings, residual)

    def to_legacy_dict(self) -> dict[str, dict]:
        """The paper's ``{"Fs": {ts: F}, "D_halves": {ts: d}}`` of pandas objects."""
        fs = {date: self.loadings.loc[date].rename_axis(None) for date in self.dates}
        d_halves = {date: self.residual_std.loc[date].rename(None) for date in self.dates}
        return {"Fs": fs, "D_halves": d_halves}


@dataclass(frozen=True)
class CovarianceHistory:
    """Full daily covariance matrices, indexed by ``(date, asset)`` with asset columns.

    Validated symmetric and positive semidefinite within relative tolerances,
    and converted to an exact factor form (``F = V sqrt(lambda)``, zero residual):
    no residual variance is added to a covariance that is already complete.
    """

    values: pd.DataFrame
    symmetry_tolerance: float = 1e-10
    psd_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        """Validate labels, symmetry, and positive semidefiniteness."""
        dates, assets = _check_panel(self.values, "Covariance history")
        check_labels(self.values.columns, "Covariance columns")
        if set(self.values.columns) != set(assets):
            raise ValueError("Covariance rows and columns must list the same assets.")
        for date in dates:
            sigma = self.values.loc[date].reindex(index=assets, columns=assets).to_numpy(float)
            scale = max(np.abs(np.diag(sigma)).max(), np.finfo(float).tiny)
            if np.abs(sigma - sigma.T).max() > self.symmetry_tolerance * scale:
                raise ValueError(f"Covariance on {date.date()} is not symmetric.")
            if np.linalg.eigvalsh(sigma).min() < -self.psd_tolerance * scale:
                raise ValueError(f"Covariance on {date.date()} is not positive semidefinite.")

    def to_factor_risk(self) -> FactorRiskHistory:
        """Exact factor form of every matrix."""
        dates, assets = _check_panel(self.values, "Covariance history")
        loadings = np.empty((len(dates), len(assets), len(assets)))
        for t, date in enumerate(dates):
            sigma = self.values.loc[date].reindex(index=assets, columns=assets).to_numpy(float)
            sigma = 0.5 * (sigma + sigma.T)
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            loadings[t] = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
        residual = np.zeros((len(dates), len(assets)))
        return FactorRiskHistory.from_arrays(dates, assets, loadings, residual)


def as_factor_risk(risk: "FactorRiskHistory | CovarianceHistory") -> FactorRiskHistory:
    """Resolve either prepared risk input to factor form."""
    if isinstance(risk, CovarianceHistory):
        return risk.to_factor_risk()
    if isinstance(risk, FactorRiskHistory):
        return risk
    raise TypeError("risk must be a FactorRiskHistory or CovarianceHistory.")


class RiskModel(Protocol):
    """Anything that turns prices into a risk history."""

    def estimate(self, prices: pd.DataFrame) -> FactorRiskHistory:
        """Risk history for the price dates it can cover."""
        ...


@dataclass(frozen=True)
class RollingSampleCovariance:
    """Sample covariance of the last ``window`` complete daily returns (the paper's estimator).

    On each date it takes the trailing ``window`` rows where all ``assets``
    have returns, computes RMS volatilities and the covariance of scaled
    returns, and uses ``F = diag(vols) @ cholesky(cov(scaled))``, so
    ``F F^T`` is the ordinary centered sample covariance. Residual std is the
    constant ``residual_std``. Columns of ``prices`` outside ``assets`` get
    zero loadings and residuals.

    ``window`` is a rolling window, not a half-life, and must exceed the number
    of assets for the Cholesky factor to exist. It suits small universes (the
    paper uses 11 days for three assets); it is not a general default.
    """

    window: int = 11
    residual_std: float = 1e-7
    assets: tuple[str, ...] | None = None

    def estimate(self, prices: pd.DataFrame) -> FactorRiskHistory:
        """Risk history from the ``window``-th return date onward."""
        assets = list(prices.columns) if self.assets is None else list(self.assets)
        check_known(assets, prices.columns, "Risk model")
        if self.window <= len(assets):
            raise ValueError(
                f"window ({self.window}) must exceed the number of assets ({len(assets)}) "
                "for a full-rank sample covariance; supply a covariance history instead."
            )
        returns = prices.pct_change().dropna(how="all")
        complete = returns[assets].dropna()
        candidates = returns.index[self.window:]
        n_complete = np.searchsorted(complete.index, candidates, side="right")
        position = {asset: j for j, asset in enumerate(prices.columns)}
        rows = [position[asset] for asset in assets]

        dates, loadings = [], []
        for date, stop in zip(candidates, n_complete, strict=True):
            if stop < self.window:
                continue
            tail = complete.iloc[stop - self.window : stop]
            vols = np.sqrt(np.square(tail).mean())
            corr = tail.div(vols, axis=1).cov()
            try:
                f_corr = np.linalg.cholesky(corr.to_numpy())
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"Sample correlation on {date.date()} is not positive definite; "
                    "check for constant or collinear returns."
                ) from exc
            F = np.zeros((len(prices.columns), len(assets)))
            F[rows] = np.diag(vols.to_numpy()) @ f_corr
            dates.append(date)
            loadings.append(F)
        residual = np.zeros((len(dates), len(prices.columns)))
        residual[:, rows] = self.residual_std
        return FactorRiskHistory.from_arrays(pd.DatetimeIndex(dates, name="date"),
                                             prices.columns, np.array(loadings), residual)


def rolling_portfolio_volatility(prices: pd.DataFrame, weights: np.ndarray,
                                 window: int = 11) -> pd.Series:
    """Rolling std of a constant-mix portfolio's daily returns (daily units).

    Missing asset returns contribute zero, as in the paper's benchmarks.
    """
    returns = prices.pct_change().dropna(how="all")
    return returns.mul(weights, axis=1).sum(axis=1).rolling(window=window).std()
