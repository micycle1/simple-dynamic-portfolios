"""Prepared market inputs, validation, and alignment to the simulator's arrays.

The public boundary is labelled pandas objects; the numerical code below it is
positional. Everything is aligned here, once, to a canonical asset order (the
price DataFrame's columns) instead of relying on independent ``.to_numpy()``
calls agreeing by coincidence.

Information timing: an observation labelled with date ``t`` is assumed usable
at the rebalance on ``t``. Nothing here infers publication times or lags data.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulator import BacktestData, RebalanceFrequency

RESERVED_LABELS = frozenset({"Cash"})  # the simulator's composition column


def check_dates(index: pd.Index, what: str) -> pd.DatetimeIndex:
    """Require a unique, increasing DatetimeIndex without NaT."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{what} must be indexed by a DatetimeIndex, got {type(index).__name__}.")
    if index.hasnans:
        raise ValueError(f"{what} has missing (NaT) dates.")
    if not index.is_unique:
        raise ValueError(f"{what} has duplicate dates, e.g. {index[index.duplicated()][0]}.")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{what} dates are not sorted.")
    return index


def check_labels(labels: pd.Index, what: str) -> pd.Index:
    """Require unique, non-missing labels."""
    if labels.hasnans:
        raise ValueError(f"{what} has missing labels.")
    if not labels.is_unique:
        raise ValueError(f"{what} has duplicate labels: {list(labels[labels.duplicated()])}.")
    return labels


def check_same_timezone(index: pd.DatetimeIndex, reference: pd.DatetimeIndex, what: str) -> None:
    """Require the same timezone convention (both naive, or the same zone)."""
    if str(index.tz) != str(reference.tz):
        raise ValueError(f"{what} timezone {index.tz} differs from the prices' {reference.tz}.")


def check_known(labels: Iterable, assets: pd.Index, what: str) -> None:
    """Reject labels outside the investable universe."""
    unknown = [label for label in labels if label not in assets]
    if unknown:
        raise ValueError(f"{what} has assets outside the investable universe: {unknown}.")


def check_complete(frame: pd.DataFrame, dates: pd.DatetimeIndex, assets: pd.Index,
                   what: str) -> pd.DataFrame:
    """Reindex ``frame`` to ``dates`` x ``assets`` and require every value to be finite."""
    check_known(frame.columns, assets, what)
    missing_assets = [asset for asset in assets if asset not in frame.columns]
    if missing_assets:
        raise ValueError(f"{what} is missing assets: {missing_assets}.")
    aligned = frame.reindex(index=dates, columns=assets)
    bad = ~np.isfinite(aligned.to_numpy(dtype=float))
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(
            f"{what} needs a finite value for every asset on every decision date; "
            f"first gap: {assets[col]} on {dates[row].date()} ({bad.any(axis=1).sum()} dates)."
        )
    return aligned


@dataclass(frozen=True)
class MarketInputs:
    """Total-return prices (dates x assets) and daily decimal cash returns.

    Prices are adjusted closes or total-return indices; NaN marks an asset
    without a price (e.g. before inception) and is allowed outside the
    simulation interval, where it only affects model warm-up. Cash returns are
    daily decimal returns (not annual rates); the observation on ``t`` accrues
    over the interval ending on ``t``.
    """

    total_return_prices: pd.DataFrame
    cash_returns: pd.Series

    def __post_init__(self) -> None:
        """Validate labels, dates, and values."""
        prices = self.total_return_prices
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("total_return_prices must be a DataFrame (dates x assets).")
        check_dates(prices.index, "total_return_prices")
        check_labels(prices.columns, "total_return_prices columns")
        reserved = RESERVED_LABELS.intersection(prices.columns)
        if reserved:
            raise ValueError(f"Asset labels {sorted(reserved)} are reserved.")
        values = prices.to_numpy(dtype=float)
        present = ~np.isnan(values)
        if not (np.isfinite(values[present]).all() and (values[present] > 0.0).all()):
            raise ValueError("total_return_prices must be positive and finite where present.")

        cash = self.cash_returns
        if isinstance(cash, pd.DataFrame):
            if cash.shape[1] != 1:
                raise ValueError("cash_returns must be a Series or a one-column DataFrame.")
            cash = cash.iloc[:, 0]
        if not isinstance(cash, pd.Series):
            raise TypeError("cash_returns must be a Series of daily decimal returns.")
        check_dates(cash.index, "cash_returns")
        check_same_timezone(cash.index, prices.index, "cash_returns")
        cash_values = cash.to_numpy(dtype=float)
        cash_present = ~np.isnan(cash_values)
        if not (np.isfinite(cash_values[cash_present]).all()
                and (cash_values[cash_present] > -1.0).all()):
            raise ValueError("cash_returns must be finite daily returns above -1.")
        object.__setattr__(self, "total_return_prices", prices.astype(float))
        object.__setattr__(self, "cash_returns", cash.astype(float))

    @property
    def assets(self) -> pd.Index:
        """Canonical asset order."""
        return self.total_return_prices.columns

    def simulation_dates(self, start=None, end=None) -> pd.DatetimeIndex:
        """Price dates in the inclusive interval ``[start, end]``."""
        dates = self.total_return_prices.loc[start:end].index
        if len(dates) < 2:
            raise ValueError(f"Fewer than two price dates between {start} and {end}.")
        return dates

    def to_backtest_data(
        self, rebalance_frequency: RebalanceFrequency, start=None, end=None
    ) -> BacktestData:
        """Validated simulator arrays for ``[start, end]``.

        Requires a complete price panel and cash returns on every simulation
        date. For the legacy missing-price behavior (unavailable assets are
        force-liquidated), call ``BacktestData.from_pandas`` directly.
        """
        dates = self.simulation_dates(start, end)
        panel = self.total_return_prices.loc[dates]
        if panel.isna().any().any():
            gaps = panel.isna()
            first = {a: gaps.index[gaps[a]][0].date() for a in panel.columns if gaps[a].any()}
            raise ValueError(
                "A complete price panel is required during the simulation interval; "
                f"first missing price per asset: {first}. Start later, drop the asset, or "
                "use BacktestData.from_pandas for legacy missing-price handling."
            )
        cash = self.cash_returns.reindex(dates)
        if cash.isna().any():
            raise ValueError(
                f"cash_returns are missing on {int(cash.isna().sum())} simulation dates, "
                f"first {cash.index[cash.isna()][0].date()}."
            )
        return BacktestData.from_pandas(
            start_date=dates[0],
            end_date=dates[-1],
            closes=self.total_return_prices,
            ffrs=cash,
            rebal_freq=rebalance_frequency,
        )


def decision_dates(data: BacktestData) -> pd.DatetimeIndex:
    """Dates on which the simulator calls the constructor (schedule plus the first day)."""
    mask = data.rebal_schedule.copy()
    mask[0] = True
    return pd.DatetimeIndex(data.timeline[mask])
