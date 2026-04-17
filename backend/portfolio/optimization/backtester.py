"""
Portfolio backtesting engine.

Provides historical performance evaluation for portfolio strategies,
including rolling-window analysis, strategy comparison, and performance
attribution.

Key metrics computed:
  - Annualised return, volatility, and Sharpe ratio
  - Maximum drawdown, Calmar ratio, Sortino ratio
  - Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
  - Rolling Sharpe and rolling drawdown analysis
  - Sector/asset attribution of returns

Typical usage::

    bt = PortfolioBacktester(risk_free_rate=0.07)
    result = bt.backtest(weights, returns_df)
    print(result["metrics"]["sharpe_ratio"])
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """Backtests portfolio strategies on historical data.

    Parameters
    ----------
    risk_free_rate : float
        Annualised risk-free rate (default ``0.07``, India 10Y bond).
    trading_days_per_year : int
        Number of trading days in a year (default ``252``).
    initial_value : float
        Starting portfolio value for equity curve (default ``1_00_000``).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.07,
        trading_days_per_year: int = 252,
        initial_value: float = 1_00_000.0,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        self.initial_value = initial_value
        self.daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        name: str = "Portfolio",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Backtest a static-weight portfolio on historical returns.

        Parameters
        ----------
        weights : dict[str, float]
            Asset -> weight mapping.  Weights are normalised internally.
        returns : pd.DataFrame
            Historical daily returns (assets in columns, dates in index).
        benchmark : pd.Series or None
            Benchmark daily returns (e.g., Nifty 50).
        name : str
            Portfolio name for identification.
        start_date, end_date : str or None
            Date filters in ``YYYY-MM-DD`` format.

        Returns
        -------
        dict
            ``{
                "name": str,
                "weights": dict,
                "metrics": dict,
                "equity_curve": pd.Series,
                "drawdown_series": pd.Series,
                "rolling_metrics": dict,
                "monthly_returns": pd.Series,
                "annual_returns": pd.Series,
                "benchmark_comparison": dict or None,
            }``
        """
        # Validate inputs
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        # Filter symbols that exist in returns
        valid_symbols = [s for s in weights if s in returns.columns]
        if not valid_symbols:
            raise ValueError("No overlap between weights keys and returns columns")

        # Filter to date range
        df = returns[valid_symbols].copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        df = df.dropna()

        if len(df) < 20:
            raise ValueError(f"Insufficient data points ({len(df)}). Need at least 20.")

        # Normalise weights
        total_w = sum(weights.get(s, 0.0) for s in valid_symbols)
        if total_w <= 0:
            raise ValueError("Total weight must be positive")
        w_dict = {s: weights.get(s, 0.0) / total_w for s in valid_symbols}

        # Compute daily portfolio returns
        w_arr = np.array([w_dict[s] for s in valid_symbols])
        port_returns = pd.Series(
            df.values @ w_arr,
            index=df.index,
            name=name,
        )

        # Equity curve
        equity = self._build_equity_curve(port_returns)

        # Drawdown series
        drawdown = self._compute_drawdown(equity)

        # Core metrics
        metrics = self._compute_full_metrics(port_returns, equity, drawdown)

        # Monthly and annual returns
        monthly = port_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        annual = port_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)

        # Rolling metrics
        rolling = self._compute_rolling_metrics(port_returns)

        # Benchmark comparison
        bench_comp = None
        if benchmark is not None:
            bench_aligned = benchmark.reindex(port_returns.index).dropna()
            if len(bench_aligned) > 20:
                bench_equity = self._build_equity_curve(bench_aligned)
                bench_dd = self._compute_drawdown(bench_equity)
                bench_metrics = self._compute_full_metrics(bench_aligned, bench_equity, bench_dd)
                bench_comp = {
                    "name": "Benchmark",
                    "metrics": bench_metrics,
                    "relative_metrics": {
                        "alpha": round(metrics["annualised_return"] - bench_metrics["annualised_return"], 4),
                        "tracking_error": self._tracking_error(port_returns, bench_aligned),
                        "information_ratio": self._information_ratio(
                            port_returns, bench_aligned
                        ),
                        "beta": self._beta(port_returns, bench_aligned),
                    },
                }

        return {
            "name": name,
            "weights": w_dict,
            "metrics": metrics,
            "equity_curve": equity,
            "drawdown_series": drawdown,
            "rolling_metrics": rolling,
            "monthly_returns": monthly,
            "annual_returns": annual,
            "benchmark_comparison": bench_comp,
            "period": {
                "start": str(df.index[0].date()),
                "end": str(df.index[-1].date()),
                "trading_days": len(df),
            },
        }

    # ------------------------------------------------------------------
    # Strategy comparison
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        strategies: dict[str, dict[str, float]],
        returns: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        """Compare multiple portfolio strategies on the same returns data.

        Parameters
        ----------
        strategies : dict[str, dict[str, float]]
            Strategy name -> weights mapping.
        returns : pd.DataFrame
            Historical returns.
        benchmark : pd.Series or None
            Benchmark returns.

        Returns
        -------
        dict
            ``{
                "strategies": {name: backtest_result, ...},
                "ranking": {metric: [names], ...},
                "correlation_matrix": pd.DataFrame,
                "best_strategy": str,
            }``
        """
        results: dict[str, dict[str, Any]] = {}
        strategy_returns: dict[str, pd.Series] = {}

        for name, weights in strategies.items():
            try:
                result = self.backtest(
                    weights, returns, benchmark=benchmark, name=name
                )
                results[name] = result

                # Collect daily returns for correlation
                valid_syms = [s for s in weights if s in returns.columns]
                total_w = sum(weights.get(s, 0.0) for s in valid_syms)
                if total_w > 0:
                    w_arr = np.array([weights.get(s, 0.0) / total_w for s in valid_syms])
                    df = returns[valid_syms].dropna()
                    strat_rets = pd.Series(df.values @ w_arr, index=df.index, name=name)
                    strategy_returns[name] = strat_rets
            except Exception as exc:
                logger.warning("Strategy '%s' backtest failed: %s", name, exc)

        if not results:
            return {"strategies": {}, "ranking": {}, "best_strategy": None}

        # Rankings by key metrics
        ranking_metrics = {
            "sharpe_ratio": ("annualised_return", False),
            "annualised_return": ("annualised_return", False),
            "annualised_volatility": ("annualised_volatility", True),
            "max_drawdown": ("max_drawdown", True),
            "sortino_ratio": ("sortino_ratio", False),
            "calmar_ratio": ("calmar_ratio", False),
        }

        rankings: dict[str, list[str]] = {}
        for metric_key, (_, lower_better) in ranking_metrics.items():
            sorted_strats = sorted(
                results.items(),
                key=lambda x: x[1]["metrics"].get(metric_key, -999 if not lower_better else 999),
                reverse=not lower_better,
            )
            rankings[metric_key] = [name for name, _ in sorted_strats]

        # Correlation matrix of strategy returns
        if len(strategy_returns) > 1:
            ret_df = pd.DataFrame(strategy_returns)
            corr_matrix = ret_df.corr()
        else:
            corr_matrix = pd.DataFrame()

        # Best strategy: highest Sharpe ratio
        best = rankings.get("sharpe_ratio", [None])[0] if rankings else None

        return {
            "strategies": results,
            "ranking": rankings,
            "correlation_matrix": corr_matrix,
            "best_strategy": best,
        }

    # ------------------------------------------------------------------
    # Rolling window analysis
    # ------------------------------------------------------------------

    def rolling_window_analysis(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        window: int = 252,
        step: int = 21,
    ) -> pd.DataFrame:
        """Rolling-window backtest with periodic rebalancing.

        Parameters
        ----------
        weights : dict[str, float]
            Static target weights.
        returns : pd.DataFrame
            Historical daily returns.
        window : int
            Rolling window size in trading days (default 1 year).
        step : int
            Step between windows in trading days (default ~1 month).

        Returns
        -------
        pd.DataFrame
            One row per window with columns:
            ``start_date, end_date, sharpe, annualised_return,
            annualised_volatility, max_drawdown, final_value``.
        """
        valid_symbols = [s for s in weights if s in returns.columns]
        if len(valid_symbols) < 2:
            return pd.DataFrame()

        total_w = sum(weights.get(s, 0.0) for s in valid_symbols)
        w_arr = np.array([weights.get(s, 0.0) / total_w for s in valid_symbols])
        df = returns[valid_symbols].dropna()

        records: list[dict[str, Any]] = []
        for i in range(0, len(df) - window + 1, step):
            window_df = df.iloc[i : i + window]
            port_ret = window_df.values @ w_arr
            port_series = pd.Series(port_ret, index=window_df.index)

            equity = (1 + port_series).cumprod()

            ann_ret = self._annualised_return(port_series)
            ann_vol = self._annualised_volatility(port_series)
            sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

            rolling_max = equity.expanding().max()
            dd = (equity - rolling_max) / rolling_max
            max_dd = float(dd.min())

            records.append({
                "start_date": str(window_df.index[0].date()),
                "end_date": str(window_df.index[-1].date()),
                "trading_days": len(window_df),
                "sharpe_ratio": round(sharpe, 4),
                "annualised_return": round(ann_ret, 4),
                "annualised_volatility": round(ann_vol, 4),
                "max_drawdown": round(max_dd, 4),
                "final_value": round(float(equity.iloc[-1]), 4),
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Performance attribution
    # ------------------------------------------------------------------

    def performance_attribution(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        sector_map: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Attribute portfolio returns to individual assets and sectors.

        Parameters
        ----------
        weights : dict[str, float]
            Portfolio weights.
        returns : pd.DataFrame
            Historical returns.
        sector_map : dict[str, str] or None
            Symbol -> sector mapping.

        Returns
        -------
        dict
            ``{
                "asset_contribution": {symbol: {return, weight, contribution}},
                "sector_contribution": {sector: {return, weight, contribution}},
                "concentration_metrics": dict,
            }``
        """
        valid_symbols = [s for s in weights if s in returns.columns]
        total_w = sum(weights.get(s, 0.0) for s in valid_symbols)
        if total_w <= 0:
            return {}

        w_dict = {s: weights.get(s, 0.0) / total_w for s in valid_symbols}
        df = returns[valid_symbols].dropna()

        # Asset-level contribution
        ann_rets = df.mean() * self.trading_days
        asset_contrib: dict[str, dict[str, float]] = {}
        for sym in valid_symbols:
            w = w_dict.get(sym, 0.0)
            r = float(ann_rets.get(sym, 0.0))
            contribution = w * r
            asset_contrib[sym] = {
                "weight": round(w, 4),
                "annualised_return": round(r, 4),
                "contribution_to_return": round(contribution, 4),
                "contribution_pct": round(contribution / sum(w_dict[s2] * float(ann_rets.get(s2, 0.0)) for s2 in valid_symbols) * 100, 2) if sum(w_dict[s2] * float(ann_rets.get(s2, 0.0)) for s2 in valid_symbols) != 0 else 0.0,
            }

        # Sector-level aggregation
        sector_contrib: dict[str, dict[str, float]] = {}
        if sector_map:
            sector_weights: dict[str, float] = {}
            sector_returns: dict[str, float] = {}
            for sym, contrib in asset_contrib.items():
                sec = sector_map.get(sym, "UNKNOWN")
                sector_weights[sec] = sector_weights.get(sec, 0.0) + contrib["weight"]
                sector_returns[sec] = sector_returns.get(sec, 0.0) + contrib["contribution_to_return"]

            for sec in sorted(sector_weights.keys()):
                sw = sector_weights[sec]
                sr = sector_returns[sec] / sw if sw > 0 else 0.0
                sector_contrib[sec] = {
                    "weight": round(sw, 4),
                    "annualised_return": round(sr, 4),
                    "contribution_to_return": round(sector_returns[sec], 4),
                }

        # Concentration metrics
        w_arr = np.array([w_dict.get(s, 0.0) for s in valid_symbols])
        hhi = float(np.sum(w_arr ** 2))
        effective_n = 1.0 / hhi if hhi > 0 else 0.0

        return {
            "asset_contribution": asset_contrib,
            "sector_contribution": sector_contrib,
            "concentration_metrics": {
                "hhi": round(hhi, 4),
                "effective_n_assets": round(effective_n, 2),
                "max_weight": round(float(w_arr.max()), 4),
                "min_nonzero_weight": round(float(w_arr[w_arr > 0.001].min()), 4) if np.any(w_arr > 0.001) else 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _build_equity_curve(self, daily_returns: pd.Series) -> pd.Series:
        """Build cumulative equity curve from daily returns."""
        return self.initial_value * (1 + daily_returns).cumprod()

    @staticmethod
    def _compute_drawdown(equity: pd.Series) -> pd.Series:
        """Compute drawdown series from equity curve."""
        rolling_max = equity.expanding().max()
        return (equity - rolling_max) / rolling_max

    def _compute_full_metrics(
        self,
        daily_returns: pd.Series,
        equity: pd.Series,
        drawdown: pd.Series,
    ) -> dict[str, float]:
        """Compute all performance metrics."""
        ann_ret = self._annualised_return(daily_returns)
        ann_vol = self._annualised_volatility(daily_returns)
        sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

        # Sortino
        excess = daily_returns - self.daily_rf
        downside = excess[excess < 0]
        downside_std = float(np.std(downside, ddof=1)) * np.sqrt(self.trading_days) if len(downside) > 1 else ann_vol
        sortino = (ann_ret - self.risk_free_rate) / downside_std if downside_std > 0 else 0.0

        # Max drawdown
        max_dd = float(drawdown.min())

        # Calmar
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        # VaR and CVaR (95% and 99%)
        var_95, cvar_95 = self._compute_var_cvar(daily_returns, 0.05)
        var_99, cvar_99 = self._compute_var_cvar(daily_returns, 0.01)

        # Best/worst day
        best_day = float(daily_returns.max())
        worst_day = float(daily_returns.min())

        # Win rate
        win_rate = float((daily_returns > 0).mean())

        # Average win / average loss
        wins = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float("inf")

        # Skewness and kurtosis
        from scipy.stats import kurtosis, skew

        ret_skew = float(skew(daily_returns.dropna(), bias=False))
        ret_kurt = float(kurtosis(daily_returns.dropna(), bias=False, fisher=True))

        return {
            "annualised_return": round(ann_ret, 4),
            "annualised_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 4),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "var_99": round(var_99, 6),
            "cvar_99": round(cvar_99, 6),
            "best_day": round(best_day, 6),
            "worst_day": round(worst_day, 6),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "profit_loss_ratio": round(profit_loss_ratio, 4),
            "skewness": round(ret_skew, 4),
            "excess_kurtosis": round(ret_kurt, 4),
            "final_value": round(float(equity.iloc[-1]), 2),
            "total_return": round(float((equity.iloc[-1] / equity.iloc[0]) - 1), 4),
            "risk_free_rate": self.risk_free_rate,
        }

    def _annualised_return(self, daily_returns: pd.Series) -> float:
        """Compute annualised return."""
        total = (1 + daily_returns).prod()
        n_years = len(daily_returns) / self.trading_days
        if n_years <= 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    @staticmethod
    def _annualised_volatility(daily_returns: pd.Series) -> float:
        """Compute annualised volatility."""
        return float(daily_returns.std(ddof=1) * np.sqrt(252))

    @staticmethod
    def _compute_var_cvar(
        daily_returns: pd.Series, alpha: float
    ) -> tuple[float, float]:
        """Compute Value at Risk and Conditional VaR (Expected Shortfall).

        Uses historical simulation method.

        Parameters
        ----------
        daily_returns : pd.Series
            Daily portfolio returns.
        alpha : float
            Significance level (e.g., 0.05 for 95% VaR).

        Returns
        -------
        (var, cvar)
        """
        sorted_returns = np.sort(daily_returns.dropna().values)
        n = len(sorted_returns)
        if n == 0:
            return 0.0, 0.0

        idx = int(n * alpha)
        var = float(sorted_returns[idx])

        # CVaR: average of returns beyond VaR
        tail = sorted_returns[:idx + 1]
        cvar = float(np.mean(tail)) if len(tail) > 0 else var

        return var, cvar

    @staticmethod
    def _compute_rolling_metrics(
        daily_returns: pd.Series,
    ) -> dict[str, pd.Series]:
        """Compute rolling Sharpe, volatility, and drawdown."""
        rolling_window = 252

        rolling_sharpe = (
            daily_returns.rolling(rolling_window).mean() * 252
        ) / (daily_returns.rolling(rolling_window).std() * np.sqrt(252))

        rolling_vol = daily_returns.rolling(rolling_window).std() * np.sqrt(252)
        rolling_ret = (1 + daily_returns).rolling(rolling_window).apply(
            lambda x: x.prod() ** (252 / len(x)) - 1 if len(x) > 0 else 0,
            raw=True,
        )

        return {
            "rolling_sharpe": rolling_sharpe,
            "rolling_volatility": rolling_vol,
            "rolling_annualised_return": rolling_ret,
        }

    @staticmethod
    def _tracking_error(
        portfolio: pd.Series, benchmark: pd.Series
    ) -> float:
        """Compute tracking error vs benchmark."""
        aligned = pd.DataFrame({
            "port": portfolio,
            "bench": benchmark,
        }).dropna()
        if len(aligned) < 20:
            return float("nan")
        excess = aligned["port"] - aligned["bench"]
        return float(excess.std(ddof=1) * np.sqrt(252))

    @staticmethod
    def _information_ratio(
        portfolio: pd.Series, benchmark: pd.Series
    ) -> float:
        """Compute information ratio (alpha / tracking error)."""
        aligned = pd.DataFrame({
            "port": portfolio,
            "bench": benchmark,
        }).dropna()
        if len(aligned) < 20:
            return float("nan")
        excess = aligned["port"] - aligned["bench"]
        te = excess.std(ddof=1) * np.sqrt(252)
        if te == 0:
            return 0.0
        ann_excess = excess.mean() * 252
        return float(ann_excess / te)

    @staticmethod
    def _beta(
        portfolio: pd.Series, benchmark: pd.Series
    ) -> float:
        """Compute portfolio beta vs benchmark."""
        aligned = pd.DataFrame({
            "port": portfolio,
            "bench": benchmark,
        }).dropna()
        if len(aligned) < 20:
            return float("nan")
        cov = aligned.cov()
        bench_var = cov.loc["bench", "bench"]
        if bench_var == 0:
            return 0.0
        return float(cov.loc["port", "bench"] / bench_var)

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        n_simulations: int = 10_000,
        n_days: int = 252,
        confidence_levels: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """Monte Carlo simulation of portfolio returns.

        Parameters
        ----------
        weights : dict[str, float]
            Asset weights.
        returns : pd.DataFrame
            Historical returns for parameter estimation.
        n_simulations : int
            Number of Monte Carlo paths.
        n_days : int
            Simulation horizon in trading days.
        confidence_levels : list[float] or None
            Percentile levels for VaR (default ``[5, 25, 50, 75, 95]``).

        Returns
        -------
        dict
            ``{
                "terminal_values": np.ndarray,
                "percentiles": {level: float},
                "prob_profitable": float,
                "expected_return": float,
                "expected_shortfall_5pct": float,
                "mean_path": np.ndarray,
            }``
        """
        valid_symbols = [s for s in weights if s in returns.columns]
        total_w = sum(weights.get(s, 0.0) for s in valid_symbols)
        if total_w <= 0:
            return {}

        w_arr = np.array([weights.get(s, 0.0) / total_w for s in valid_symbols])
        df = returns[valid_symbols].dropna()

        # Estimate parameters
        mu = (df.mean().values @ w_arr)  # Daily expected return
        cov = df.cov().values
        port_var = float(w_arr @ cov @ w_arr)
        port_std = np.sqrt(port_var)

        # Simulate
        rng = np.random.default_rng(42)
        z = rng.standard_normal((n_simulations, n_days))

        # Cholesky for correlated returns
        try:
            L = np.linalg.cholesky(cov)
            correlated = z @ L.T
            sim_returns = correlated @ w_arr
        except np.linalg.LinAlgError:
            # Fallback: univariate normal
            sim_returns = mu + port_std * z

        # Cumulative returns
        cum_returns = np.cumprod(1 + sim_returns, axis=1)
        terminal_values = self.initial_value * cum_returns[:, -1]

        # Confidence levels
        levels = confidence_levels or [5, 25, 50, 75, 95]
        percentiles = {}
        for level in levels:
            percentiles[level] = float(np.percentile(terminal_values, level))

        # Metrics
        prob_profitable = float(np.mean(terminal_values > self.initial_value))
        expected_terminal = float(np.mean(terminal_values))
        es_5 = float(np.mean(terminal_values[terminal_values <= np.percentile(terminal_values, 5)]))

        # Mean path
        mean_path = self.initial_value * np.mean(cum_returns, axis=0)

        return {
            "n_simulations": n_simulations,
            "n_days": n_days,
            "terminal_values": terminal_values,
            "percentiles": percentiles,
            "prob_profitable": round(prob_profitable, 4),
            "expected_terminal_value": round(expected_terminal, 2),
            "expected_shortfall_5pct": round(es_5, 2),
            "mean_path": mean_path,
            "annualised_mean_return": round(float(mu * 252), 4),
            "annualised_volatility": round(float(port_std * np.sqrt(252)), 4),
        }
