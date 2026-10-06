"""Exponentially weighted ridge regression of forward returns on features."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adapters import AlphaForecasts


def ridge_forecast_history(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    assets: Sequence[str] | None = None,
    horizon_days: int = 100,
    max_history: int = 512,
    min_window: int = 63,
    halflife: float = 252,
    ridge: float = 10.0,
    periods_per_year: int = 252,
    verbose: bool = False,
) -> pd.DataFrame:
    """Annualized forward-return forecasts for every price date.

    On each date ``t`` the target is ``(periods_per_year / h) * (P[t+h] / P[t] - 1)``,
    and only targets matured by ``t`` are used: up to ``max_history`` trailing
    complete observations, at least ``min_window``, with exponential weights of
    half-life ``halflife`` (newest largest). Features are scaled by their trailing
    unweighted standard deviations, not centered, and no intercept is fitted.
    Feature columns incomplete over the window or on ``t`` are dropped for that
    fit. Dates with insufficient history get zero forecasts.

    ``features`` may have arbitrary column names but must share ``prices``'
    index. Columns of ``prices`` not in ``assets`` receive zero forecasts.
    """
    if not prices.index.equals(features.index):
        raise ValueError("Prices and features must have the same index.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if max_history <= 0:
        raise ValueError("max_history must be positive.")
    if halflife <= 0:
        raise ValueError("halflife must be positive.")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative.")
    if min_window <= 0:
        raise ValueError("min_window must be positive.")

    if assets is None:
        assets = list(prices.columns)
    assets = list(assets)
    missing_assets = [asset for asset in assets if asset not in prices.columns]
    if missing_assets:
        raise ValueError(f"Assets not present in prices: {missing_assets}")

    h = horizon_days
    features = features.replace([np.inf, -np.inf], np.nan)
    targets = prices[assets].pct_change(h, fill_method=None).shift(-h) * periods_per_year / h
    targets = targets.replace([np.inf, -np.inf], np.nan)

    out = fit_ridge_path(
        features.to_numpy(dtype=np.float64),
        targets.to_numpy(dtype=np.float64),
        horizon_days=h,
        max_history=max_history,
        min_window=min_window,
        halflife=halflife,
        ridge=ridge,
        verbose_index=targets.index if verbose else None,
    )
    alphas = pd.DataFrame(out, index=targets.index, columns=targets.columns)
    return alphas.reindex(columns=prices.columns, fill_value=0.0)


def fit_ridge_path(
    X: np.ndarray,
    Y: np.ndarray,
    horizon_days: int,
    max_history: int,
    min_window: int,
    halflife: float,
    ridge: float,
    verbose_index: pd.Index | None = None,
) -> np.ndarray:
    """Walk-forward ridge predictions: row ``i`` of ``X`` predicts row ``i`` of ``Y``.

    ``Y[j]`` is only observable ``horizon_days`` rows after ``j``.
    """
    h = horizon_days
    n_time, _ = Y.shape

    # run_len[j] is the number of consecutive complete all-asset target rows
    # ending at row j, inclusive.
    valid = ~np.isnan(Y).any(axis=1)
    run_len = np.zeros(n_time, dtype=int)

    for j in range(n_time):
        if not valid[j]:
            run_len[j] = 0
        elif j == 0:
            run_len[j] = 1
        else:
            run_len[j] = run_len[j - 1] + 1

    # Each window takes a trailing slice, so its newest observation receives
    # the largest weight.
    decay = np.log(2.0) / halflife
    ages = np.arange(max_history - 1, -1, -1)
    w_full = np.exp(-decay * ages)

    out = np.zeros_like(Y)

    for i in range(min_window + h - 1, n_time):
        # Targets in [lo, hi) are fully observable by prediction day i.
        hi = i + 1 - h

        if hi <= 0:
            continue

        window = min(run_len[hi - 1], max_history)

        if window < min_window:
            continue

        lo = hi - window

        Yw = Y[lo:hi]
        Xw = X[lo:hi]
        x_now = X[i]

        # Keep features observed throughout the window and on prediction day.
        ft_mask = ~np.isnan(Xw).any(axis=0) & ~np.isnan(x_now)

        if not ft_mask.any():
            continue

        feature_scale = np.maximum(np.std(Xw[:, ft_mask], axis=0), 1e-12)
        X_fit = Xw[:, ft_mask] / feature_scale
        x_pred = x_now[ft_mask] / feature_scale

        row_w = w_full[-window:].copy()
        row_w /= row_w.sum()

        # Exponentially weighted feature Gram and feature-target cross moment.
        gram = X_fit.T @ (row_w[:, None] * X_fit)
        cross = X_fit.T @ (row_w[:, None] * Yw)

        penalty = ridge * np.eye(X_fit.shape[1])

        try:
            beta = np.linalg.solve(gram + penalty, cross)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(gram + penalty, cross, rcond=None)[0]

        out[i] = x_pred @ beta

        if verbose_index is not None and i % 500 == 0:
            print(f"{verbose_index[i].date()}: alpha = {np.round(out[i], 4)}")

    return out


@dataclass(frozen=True)
class RidgeAlpha:
    """Ridge alpha model; ``forecast`` returns annualized forecasts."""

    horizon_days: int = 100
    max_history: int = 512
    min_window: int = 63
    halflife: float = 252
    ridge: float = 10.0
    periods_per_year: int = 252
    assets: tuple[str, ...] | None = None

    def forecast(
        self, prices: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> AlphaForecasts:
        """Forecast history for every price date."""
        if features is None:
            raise ValueError("RidgeAlpha needs a features DataFrame.")
        values = ridge_forecast_history(
            prices,
            features,
            assets=self.assets,
            horizon_days=self.horizon_days,
            max_history=self.max_history,
            min_window=self.min_window,
            halflife=self.halflife,
            ridge=self.ridge,
            periods_per_year=self.periods_per_year,
        )
        return AlphaForecasts.from_annualized(values, self.periods_per_year)
