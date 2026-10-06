"""Static strategic baseline and optional initial portfolio.

These are independent concepts: the baseline is the strategic reference the
optimizer is anchored to; the initial portfolio is the state the simulation
starts from. Neither is inferred from the other.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import check_known, check_labels

DEFAULT_TOLERANCE = 1e-8


def _weights_series(values, what: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{what} must be a pandas Series indexed by asset.")
    check_labels(values.index, what)
    values = values.astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{what} must be finite.")
    if (values < 0.0).any():
        raise ValueError(f"{what} must be nonnegative: {values[values < 0.0].to_dict()}.")
    return values


def _aligned(values: pd.Series, assets: pd.Index, what: str) -> np.ndarray:
    check_known(values.index, assets, what)
    return values.reindex(assets, fill_value=0.0).to_numpy(dtype=float)


@dataclass(frozen=True)
class StaticBaseline:
    """Risky-sleeve proportions summing to one; cash is not an entry.

    Investable assets omitted from ``weights`` are zero-baseline assets: the
    optimizer may still allocate to them within the anchor's trust region.
    Malformed input is rejected, never silently normalized; see ``normalized``.
    """

    weights: pd.Series
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        """Validate the weights."""
        weights = _weights_series(self.weights, "Baseline weights")
        total = weights.sum()
        if abs(total - 1.0) > self.tolerance:
            raise ValueError(
                f"Baseline weights must sum to one (got {total:.12g}); cash is not a baseline "
                "entry. Use StaticBaseline.normalized for holdings-derived weights."
            )
        object.__setattr__(self, "weights", weights)

    @classmethod
    def normalized(cls, risky_values: pd.Series) -> "StaticBaseline":
        """Explicitly normalize nonnegative risky holdings (cash already removed)."""
        risky_values = _weights_series(risky_values, "Risky holdings")
        total = risky_values.sum()
        if not total > 0.0:
            raise ValueError("Risky holdings must have a positive total.")
        return cls(risky_values / total)

    def aligned(self, assets: pd.Index) -> np.ndarray:
        """Weights in ``assets`` order; unknown labels are an error."""
        return _aligned(self.weights, assets, "Baseline")


@dataclass(frozen=True)
class InitialPortfolio:
    """Starting risky weights and cash weight, as fractions of starting NAV.

    Holdings are market values, not share counts. There is no tax basis.
    """

    asset_weights: pd.Series
    cash_weight: float
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        """Validate that risky and cash weights form a whole portfolio."""
        weights = _weights_series(self.asset_weights, "Initial asset weights")
        if not np.isfinite(self.cash_weight):
            raise ValueError("Initial cash weight must be finite.")
        total = weights.sum() + self.cash_weight
        if abs(total - 1.0) > self.tolerance:
            raise ValueError(f"Initial asset and cash weights must sum to one (got {total:.12g}).")
        object.__setattr__(self, "asset_weights", weights)
        object.__setattr__(self, "cash_weight", float(self.cash_weight))

    @classmethod
    def from_weights(cls, asset_weights: pd.Series, cash_weight: float) -> "InitialPortfolio":
        """Portfolio from weights that, with cash, sum to one."""
        return cls(asset_weights, cash_weight)

    @classmethod
    def from_values(cls, asset_values: pd.Series, cash_value: float) -> "InitialPortfolio":
        """Portfolio from market values of holdings and cash."""
        asset_values = _weights_series(asset_values, "Initial asset values")
        total = asset_values.sum() + cash_value
        if not total > 0.0:
            raise ValueError("Initial portfolio must have positive total value.")
        return cls(asset_values / total, cash_value / total)

    def aligned(self, assets: pd.Index) -> np.ndarray:
        """Risky weights in ``assets`` order; unknown labels are an error."""
        return _aligned(self.asset_weights, assets, "Initial portfolio")
