"""Performance metrics for backtest results.

Conventions (kept from the paper): returns are compounded annual growth rates
(CAGR) over ``periods_per_year``; "Return - <cash>" is the compounded CAGR in
excess of the cash reference; the paper's cash-adjusted Sharpe ratio,
"Sharpe Ratio (<cash>)", is that geometric excess CAGR divided by annualized
volatility, not the usual arithmetic Sharpe ratio (see ``arithmetic_sharpe``).
"""

import numpy as np
import pandas as pd

from sbg_portfolios.simulator import BacktestResults


def consistency(navs: pd.Series, periods_per_year: int = 252) -> float:
    """Consistency metric kappa from the cumulative log-return trajectory."""
    L = np.log(navs.to_numpy() / navs.to_numpy()[0])
    t = np.arange(len(L), dtype=float)
    b = L[-1] / t[-1]  # mean per-period log return
    return (periods_per_year * b * len(L)) / np.abs(L - b * t).sum()


def mdcr(navs: pd.Series) -> float:
    """Mean deviation from constant return (MDCR) from the cumulative log-return trajectory."""
    L = np.log(navs.to_numpy() / navs.to_numpy()[0])
    rho = L[-1] / (len(L))  # mean per-period log return
    t = np.arange(len(L), dtype=float)
    exp_nav = np.exp(rho * t)
    dcrs = np.abs(navs / exp_nav - 1)
    return dcrs.mean()


def geometric_excess_sharpe(navs: pd.Series, cash_reference: pd.Series,
                            periods_per_year: int = 252) -> float:
    """The paper's cash-adjusted Sharpe: compounded excess CAGR over annualized volatility."""
    returns = navs.pct_change().dropna()
    growth = (navs.iloc[-1] / navs.iloc[0]) / (1 + cash_reference.loc[returns.index]).prod()
    excess_cagr = growth ** (periods_per_year / len(returns)) - 1
    return excess_cagr / (returns.std() * np.sqrt(periods_per_year))


def arithmetic_sharpe(navs: pd.Series, cash_reference: pd.Series,
                      periods_per_year: int = 252) -> float:
    """Conventional Sharpe: annualized mean daily excess return over its annualized std."""
    returns = navs.pct_change().dropna()
    excess = returns - cash_reference.loc[returns.index]
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def compute_metrics(
    results: BacktestResults,
    cash_reference: pd.Series | None = None,
    inflation: pd.Series | None = None,
    *,
    periods_per_year: int = 252,
    cash_label: str = "FFR",
    inflation_label: str = "CPI",
) -> pd.Series:
    """Compute metrics for backtest results.

    ``cash_reference`` (daily decimal returns) and ``inflation`` (daily decimal
    inflation) are optional evaluation references, independent of the cash the
    strategy actually earned; rows depending on an omitted reference are left
    out. The labels name those rows (the paper used FFR and CPI).
    """
    navs = results.navs
    returns = navs.pct_change().dropna()
    cumu_rets = navs.iloc[-1] / navs.iloc[0]
    n = len(returns)

    mean_returns = cumu_rets ** (periods_per_year / n) - 1
    vol = returns.std() * np.sqrt(periods_per_year)

    cummax = navs.cummax()
    drawdown = (cummax - navs) / cummax

    stats = {"Return": mean_returns}
    if cash_reference is not None:
        cumu_cash = (1 + cash_reference.loc[returns.index]).prod()
        stats[f"Return - {cash_label}"] = (cumu_rets / cumu_cash) ** (periods_per_year / n) - 1
    if inflation is not None:
        cumu_cpi = (1 + inflation.loc[returns.index]).prod()
        stats[f"Return - {inflation_label}"] = (cumu_rets / cumu_cpi) ** (periods_per_year / n) - 1
    stats["Volatility"] = vol
    stats["Sharpe Ratio"] = mean_returns / vol
    if cash_reference is not None:
        stats[f"Sharpe Ratio ({cash_label})"] = stats[f"Return - {cash_label}"] / vol
    if inflation is not None:
        stats[f"Sharpe Ratio ({inflation_label})"] = stats[f"Return - {inflation_label}"] / vol
    stats["Max Drawdown"] = drawdown.max()
    stats["Avg. Drawdown"] = drawdown.mean()
    stats["Turnover"] = results.turnover.mean() * periods_per_year
    stats["Consistency"] = consistency(navs, periods_per_year)
    stats["MDCR"] = mdcr(navs)
    return pd.Series(stats)
