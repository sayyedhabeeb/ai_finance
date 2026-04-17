"""
Portfolio optimisation using multiple strategies.

Implements four core optimisation approaches backed by PyPortfolioOpt:
  - Mean-Variance (Markowitz)
  - Black-Litterman
  - Risk Parity (equal risk contribution)
  - Hierarchical Risk Parity (HRP)

All strategies produce clean weight dictionaries and standardised
performance metrics suitable for downstream consumption.

Typical usage::

    opt = PortfolioOptimizer()
    result = opt.optimize_mean_variance(returns_df, risk_free_rate=0.07)
    print(result["weights"])      # {RELIANCE.NS: 0.15, TCS.NS: 0.12, ...}
    print(result["metrics"])      # {sharpe: 1.82, volatility: 0.18, ...}
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress PyPortfolioOpt verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypfopt")


class PortfolioOptimizer:
    """Portfolio optimisation using multiple strategies.

    Parameters
    ----------
    risk_free_rate : float
        Annualised risk-free rate (default ``0.07``, approximating the
        India 10Y government bond yield).
    max_weight : float
        Maximum weight for any single asset (default ``0.10`` for SEBI
        mutual-fund style constraint).
    min_weight : float
        Minimum non-zero weight (default ``0.01``).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.07,
        max_weight: float = 0.10,
        min_weight: float = 0.01,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _validate_returns(self, returns: pd.DataFrame) -> None:
        """Validate that the returns DataFrame is usable for optimisation."""
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        if len(returns.columns) < 2:
            raise ValueError("Need at least 2 assets for optimisation")
        if returns.isna().all().any():
            raise ValueError("One or more assets have all-NaN returns")

    def _clean_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Clean returns: forward-fill, drop columns that are mostly NaN."""
        # Forward-fill up to 3 days
        cleaned = returns.ffill(limit=3).bfill(limit=1)
        # Drop columns with > 10 % NaN after cleaning
        threshold = int(len(cleaned) * 0.9)
        cleaned = cleaned.dropna(axis=1, thresh=threshold)
        # Drop any remaining NaN rows
        cleaned = cleaned.dropna()
        return cleaned

    def _compute_metrics(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        expected_returns_vec: Optional[pd.Series] = None,
    ) -> dict[str, float]:
        """Compute portfolio performance metrics.

        Parameters
        ----------
        weights : dict
            Asset -> weight mapping.
        returns : pd.DataFrame
            Historical returns.
        expected_returns_vec : pd.Series or None
            Expected return vector.  If ``None``, uses historical mean.

        Returns
        -------
        dict
            ``{
                annualised_return, annualised_volatility, sharpe_ratio,
                sortino_ratio, max_drawdown, calmar_ratio,
                weights_concentration (HHI), effective_n_assets
            }``
        """
        from pypfopt import risk_models

        w_arr = np.array([weights.get(col, 0.0) for col in returns.columns])
        w_arr = w_arr / w_arr.sum()  # Normalise to 1.0

        if expected_returns_vec is None:
            mu = returns.mean() * 252  # Annualised
        else:
            mu = expected_returns_vec

        # Expected portfolio return
        port_return = float(np.dot(w_arr, mu))

        # Covariance matrix and portfolio volatility
        cov = risk_models.sample_cov(returns)
        port_vol = float(np.sqrt(w_arr @ cov.values @ w_arr) * np.sqrt(252))

        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        # Sortino ratio (downside risk)
        excess = returns.values @ w_arr
        downside = excess[excess < 0]
        downside_std = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else port_vol
        sortino = (port_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0.0

        # Max drawdown
        cumulative = (1 + pd.Series(excess, index=returns.index)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())

        # Calmar ratio
        calmar = port_return / abs(max_dd) if max_dd != 0 else 0.0

        # Herfindahl-Hirschman Index (concentration)
        w_sq = w_arr ** 2
        hhi = float(np.sum(w_sq))

        # Effective number of assets
        effective_n = float(1.0 / hhi) if hhi > 0 else 0.0

        return {
            "annualised_return": round(port_return, 4),
            "annualised_volatility": round(port_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 4),
            "hhi_concentration": round(hhi, 4),
            "effective_n_assets": round(effective_n, 2),
            "risk_free_rate": self.risk_free_rate,
            "n_assets": int((w_arr > 0.005).sum()),
        }

    # ------------------------------------------------------------------
    # Mean-Variance Optimisation (Markowitz)
    # ------------------------------------------------------------------

    def optimize_mean_variance(
        self,
        returns: pd.DataFrame,
        method: str = "max_sharpe",
        risk_free_rate: Optional[float] = None,
        sector_limits: Optional[dict[str, float]] = None,
        sector_map: Optional[dict[str, str]] = None,
        max_weight: Optional[float] = None,
        min_weight: Optional[float] = None,
    ) -> dict[str, Any]:
        """Mean-Variance (Markowitz) portfolio optimisation.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical daily returns (assets in columns).
        method : str
            ``"max_sharpe"`` (maximise Sharpe ratio) or ``"min_volatility"``.
        risk_free_rate : float or None
            Override the instance default.
        sector_limits : dict[str, float] or None
            Sector -> max total weight (e.g. ``{"BANKING": 0.25}``).
        sector_map : dict[str, str] or None
            Symbol -> sector mapping.  Required if *sector_limits* is set.
        max_weight : float or None
            Override max single-asset weight.
        min_weight : float or None
            Override min non-zero weight.

        Returns
        -------
        dict
            ``{
                "strategy": "mean_variance",
                "method": str,
                "weights": {symbol: float, ...},
                "metrics": {...},
                "cleaned_assets": list[str],
            }``
        """
        from pypfopt import (
            CLA,
            BlackLittermanModel,
            EfficientFrontier,
            expected_returns,
            risk_models,
        )

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        mw = max_weight if max_weight is not None else self.max_weight
        miw = min_weight if min_weight is not None else self.min_weight

        self._validate_returns(returns)
        cleaned = self._clean_returns(returns)

        mu = expected_returns.mean_historical_return(cleaned)
        S = risk_models.sample_cov(cleaned)

        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)  # Long-only

        # Weight bounds
        ef.add_constraint(lambda w: w <= mw)

        # Sector constraints via linear constraints
        if sector_limits and sector_map:
            for sector, limit in sector_limits.items():
                sector_indices = [
                    i for i, sym in enumerate(cleaned.columns)
                    if sector_map.get(sym) == sector
                ]
                if sector_indices:
                    ef.add_constraint(
                        lambda w, idx=sector_indices: np.sum(w[idx]) <= limit
                    )

        if method == "max_sharpe":
            ef.max_sharpe(risk_free_rate=rf)
        elif method == "min_volatility":
            ef.min_volatility()
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'max_sharpe' or 'min_volatility'.")

        # Clean small weights
        cleaned_weights = ef.clean_weights(cutoff=miw, rounding=4)
        ef.portfolio_performance(verbose=False)

        weights_dict = {col: cleaned_weights.get(col, 0.0) for col in cleaned.columns}

        # Compute full metrics
        metrics = self._compute_metrics(weights_dict, cleaned, mu)
        metrics["method"] = method

        return {
            "strategy": "mean_variance",
            "method": method,
            "weights": weights_dict,
            "metrics": metrics,
            "cleaned_assets": list(cleaned.columns),
        }

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def optimize_black_litterman(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series,
        views: dict[str, float],
        view_confidences: Optional[dict[str, float]] = None,
        risk_free_rate: Optional[float] = None,
        tau: float = 0.05,
        delta: Optional[float] = None,
        sector_limits: Optional[dict[str, float]] = None,
        sector_map: Optional[dict[str, str]] = None,
        max_weight: Optional[float] = None,
        min_weight: Optional[float] = None,
    ) -> dict[str, Any]:
        """Black-Litterman portfolio optimisation.

        Combines market equilibrium with investor views to produce
        more stable expected returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical daily returns.
        market_caps : pd.Series
            Market capitalisation per asset (index = symbol).
        views : dict[str, float]
            Investor views: ``{symbol: expected_return}`` or
            ``{symbol: relative_view}``.  Positive means bullish.
        view_confidences : dict[str, float] or None
            Confidence per view (0-1).  Higher = more conviction.
        risk_free_rate : float or None
            Override default risk-free rate.
        tau : float
            Uncertainty parameter (scalar, default ``0.05``).
        delta : float or None
            Risk aversion coefficient.  Default is
            ``(market_return - rf) / market_variance``.
        sector_limits, sector_map, max_weight, min_weight :
            Same as :meth:`optimize_mean_variance`.

        Returns
        -------
        dict
            ``{
                "strategy": "black_litterman",
                "implied_returns": dict,
                "adjusted_returns": dict,
                "weights": dict,
                "metrics": dict,
            }``
        """
        from pypfopt import (
            BlackLittermanModel,
            EfficientFrontier,
            expected_returns,
            risk_models,
        )

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        mw = max_weight if max_weight is not None else self.max_weight
        miw = min_weight if min_weight is not None else self.min_weight

        self._validate_returns(returns)
        cleaned = self._clean_returns(returns)

        # Align market caps with cleaned returns columns
        common = [s for s in cleaned.columns if s in market_caps.index]
        if len(common) < 2:
            raise ValueError("Insufficient overlap between returns columns and market_caps index")
        cleaned = cleaned[common]
        mcaps = market_caps.loc[common]

        # Market equilibrium returns
        S = risk_models.sample_cov(cleaned)
        market_prior = BlackLittermanModel.market_implied_prior_returns(
            mcaps, delta=delta, risk_free_rate=rf, prior=S
        )

        # Build view matrix (P) and view vector (Q)
        view_assets = [s for s in views.keys() if s in cleaned.columns]
        if not view_assets:
            raise ValueError("No view assets found in returns columns")

        # Absolute views
        P = np.zeros((len(view_assets), len(cleaned.columns)))
        Q = np.array([views[s] for s in view_assets])

        for i, sym in enumerate(view_assets):
            j = list(cleaned.columns).index(sym)
            P[i, j] = 1.0

        # View confidences (Omega matrix)
        if view_confidences:
            omega_diag = []
            for s in view_assets:
                conf = view_confidences.get(s, 0.5)
                # Higher confidence -> smaller variance
                # Scale tau * asset_var / confidence
                var = S.loc[s, s] if s in S.index else 0.001
                omega_diag.append(tau * var / max(conf, 0.01))
            omega = np.diag(omega_diag)
        else:
            omega = None

        bl = BlackLittermanModel(
            S,
            pi=market_prior,
            Q=Q,
            P=P,
            tau=tau,
            omega=omega,
        )

        # Get posterior returns
        bl_returns = bl.bl_returns()

        # Optimise on BL returns
        ef = EfficientFrontier(bl_returns, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: w <= mw)

        if sector_limits and sector_map:
            for sector, limit in sector_limits.items():
                sector_indices = [
                    i for i, sym in enumerate(cleaned.columns)
                    if sector_map.get(sym) == sector
                ]
                if sector_indices:
                    ef.add_constraint(
                        lambda w, idx=sector_indices: np.sum(w[idx]) <= limit
                    )

        ef.max_sharpe(risk_free_rate=rf)
        cleaned_weights = ef.clean_weights(cutoff=miw, rounding=4)

        weights_dict = {col: cleaned_weights.get(col, 0.0) for col in cleaned.columns}
        implied_dict = {col: float(market_prior.get(col, 0.0)) for col in cleaned.columns}
        adjusted_dict = {col: float(bl_returns.get(col, 0.0)) for col in cleaned.columns}

        metrics = self._compute_metrics(weights_dict, cleaned, bl_returns)

        return {
            "strategy": "black_litterman",
            "views_applied": {s: views[s] for s in view_assets},
            "implied_returns": implied_dict,
            "adjusted_returns": adjusted_dict,
            "weights": weights_dict,
            "metrics": metrics,
            "cleaned_assets": list(cleaned.columns),
        }

    # ------------------------------------------------------------------
    # Risk Parity
    # ------------------------------------------------------------------

    def optimize_risk_parity(
        self,
        returns: pd.DataFrame,
        risk_free_rate: Optional[float] = None,
        objective: str = "sharpe",
        sector_limits: Optional[dict[str, float]] = None,
        sector_map: Optional[dict[str, str]] = None,
        max_weight: Optional[float] = None,
    ) -> dict[str, Any]:
        """Risk Parity / Equal Risk Contribution optimisation.

        Allocates capital so that each asset contributes equally to
        portfolio risk.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical daily returns.
        objective : str
            ``"sharpe"`` or ``"volatility"``.
        risk_free_rate, sector_limits, sector_map, max_weight :
            Same as :meth:`optimize_mean_variance`.

        Returns
        -------
        dict
            ``{
                "strategy": "risk_parity",
                "weights": dict,
                "risk_contributions": dict,
                "metrics": dict,
            }``
        """
        from pypfopt import (
            EfficientFrontier,
            expected_returns,
            risk_models,
        )

        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        mw = max_weight if max_weight is not None else self.max_weight

        self._validate_returns(returns)
        cleaned = self._clean_returns(returns)

        mu = expected_returns.mean_historical_return(cleaned)
        S = risk_models.sample_cov(cleaned)

        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: w <= mw)

        if sector_limits and sector_map:
            for sector, limit in sector_limits.items():
                sector_indices = [
                    i for i, sym in enumerate(cleaned.columns)
                    if sector_map.get(sym) == sector
                ]
                if sector_indices:
                    ef.add_constraint(
                        lambda w, idx=sector_indices: np.sum(w[idx]) <= limit
                    )

        if objective == "sharpe":
            ef.max_sharpe(risk_free_rate=rf)
        else:
            ef.min_volatility()

        cleaned_weights = ef.clean_weights(cutoff=0.005, rounding=4)
        weights_dict = {col: cleaned_weights.get(col, 0.0) for col in cleaned.columns}

        # Compute risk contributions
        w_arr = np.array([weights_dict.get(col, 0.0) for col in cleaned.columns])
        w_arr = w_arr / w_arr.sum()
        cov = S.values
        port_var = w_arr @ cov @ w_arr
        marginal_risk = cov @ w_arr
        risk_contrib = w_arr * marginal_risk
        if port_var > 0:
            risk_contrib_pct = risk_contrib / port_var
        else:
            risk_contrib_pct = np.ones_like(risk_contrib) / len(risk_contrib)

        rc_dict = {
            col: round(float(risk_contrib_pct[i]), 4)
            for i, col in enumerate(cleaned.columns)
        }

        metrics = self._compute_metrics(weights_dict, cleaned, mu)
        metrics["objective"] = objective

        return {
            "strategy": "risk_parity",
            "weights": weights_dict,
            "risk_contributions": rc_dict,
            "metrics": metrics,
            "cleaned_assets": list(cleaned.columns),
        }

    # ------------------------------------------------------------------
    # Hierarchical Risk Parity (HRP)
    # ------------------------------------------------------------------

    def optimize_hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        linkage_method: str = "single",
        max_weight: Optional[float] = None,
    ) -> dict[str, Any]:
        """Hierarchical Risk Parity optimisation.

        Uses a hierarchical clustering approach to build a diversified
        portfolio without requiring invertibility of the covariance matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical daily returns.
        linkage_method : str
            SciPy linkage method: ``"single"``, ``"average"``, ``"complete"``,
            ``"ward"``.
        max_weight : float or None
            Post-hoc cap on any single weight.

        Returns
        -------
        dict
            ``{
                "strategy": "hierarchical_risk_parity",
                "weights": dict,
                "cluster_order": list[str],
                "metrics": dict,
            }``
        """
        from pypfopt import HRPOpt

        self._validate_returns(returns)
        cleaned = self._clean_returns(returns)

        hrp = HRPOpt(cleaned)
        raw_weights = hrp.optimize(linkage_method=linkage_method)

        # Convert to dict
        weights_dict = {col: float(raw_weights.get(col, 0.0)) for col in cleaned.columns}

        # Apply max_weight cap if specified
        mw = max_weight if max_weight is not None else self.max_weight
        if mw < 1.0:
            weights_dict = self._apply_weight_cap(weights_dict, mw)

        # Get cluster order (dendrogram leaves)
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            corr = cleaned.corr()
            dist = np.sqrt((1 - corr) / 2)
            condensed = dist.where(np.triu(np.ones(dist.shape), k=1).astype(bool)).stack().values
            Z = linkage(condensed, method=linkage_method)
            cluster_order = [cleaned.columns[i] for i in leaves_list(Z)]
        except Exception:
            cluster_order = list(cleaned.columns)

        metrics = self._compute_metrics(weights_dict, cleaned)

        return {
            "strategy": "hierarchical_risk_parity",
            "weights": weights_dict,
            "cluster_order": cluster_order,
            "linkage_method": linkage_method,
            "metrics": metrics,
            "cleaned_assets": list(cleaned.columns),
        }

    @staticmethod
    def _apply_weight_cap(
        weights: dict[str, float], max_weight: float
    ) -> dict[str, float]:
        """Cap individual weights and redistribute proportionally."""
        capped = {}
        total_capped = 0.0
        for sym, w in weights.items():
            if w > max_weight:
                capped[sym] = max_weight
                total_capped += w - max_weight
            else:
                capped[sym] = w

        # Redistribute excess proportionally to non-capped assets
        non_capped = {s: w for s, w in capped.items() if w < max_weight}
        non_capped_total = sum(non_capped.values())

        if non_capped_total > 0 and total_capped > 0:
            for sym in non_capped:
                capped[sym] += (non_capped[sym] / non_capped_total) * total_capped

        # Normalise to sum to 1.0
        total = sum(capped.values())
        if total > 0:
            capped = {s: w / total for s, w in capped.items()}

        return capped

    # ------------------------------------------------------------------
    # Multi-strategy comparison
    # ------------------------------------------------------------------

    def compare_strategies(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[pd.Series] = None,
        views: Optional[dict[str, float]] = None,
        sector_limits: Optional[dict[str, float]] = None,
        sector_map: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Run all strategies and return a comparison.

        Parameters
        ----------
        returns, market_caps, views, sector_limits, sector_map :
            See individual strategy methods.

        Returns
        -------
        dict
            ``{
                "strategies": {name: result, ...},
                "best_sharpe": str,
                "best_volatility": str,
                "rankings": {metric: [ranked_names], ...},
            }``
        """
        results: dict[str, dict[str, Any]] = {}

        # 1. Mean-Variance (Max Sharpe)
        try:
            results["mean_variance_sharpe"] = self.optimize_mean_variance(
                returns, method="max_sharpe",
                sector_limits=sector_limits, sector_map=sector_map,
            )
        except Exception as exc:
            logger.warning("Mean-Variance (Sharpe) failed: %s", exc)

        # 2. Mean-Variance (Min Volatility)
        try:
            results["mean_variance_minvol"] = self.optimize_mean_variance(
                returns, method="min_volatility",
                sector_limits=sector_limits, sector_map=sector_map,
            )
        except Exception as exc:
            logger.warning("Mean-Variance (MinVol) failed: %s", exc)

        # 3. Black-Litterman
        if market_caps and views:
            try:
                results["black_litterman"] = self.optimize_black_litterman(
                    returns, market_caps=market_caps, views=views,
                    sector_limits=sector_limits, sector_map=sector_map,
                )
            except Exception as exc:
                logger.warning("Black-Litterman failed: %s", exc)

        # 4. Risk Parity
        try:
            results["risk_parity"] = self.optimize_risk_parity(
                returns, sector_limits=sector_limits, sector_map=sector_map,
            )
        except Exception as exc:
            logger.warning("Risk Parity failed: %s", exc)

        # 5. HRP
        try:
            results["hrp"] = self.optimize_hierarchical_risk_parity(returns)
        except Exception as exc:
            logger.warning("HRP failed: %s", exc)

        # Rankings
        sharpe_ranking = sorted(
            results.items(),
            key=lambda x: x[1]["metrics"].get("sharpe_ratio", -999),
            reverse=True,
        )
        vol_ranking = sorted(
            results.items(),
            key=lambda x: x[1]["metrics"].get("annualised_volatility", 999),
        )

        return {
            "strategies": results,
            "best_sharpe": sharpe_ranking[0][0] if sharpe_ranking else None,
            "best_volatility": vol_ranking[0][0] if vol_ranking else None,
            "rankings": {
                "sharpe_ratio": [name for name, _ in sharpe_ranking],
                "volatility": [name for name, _ in vol_ranking],
            },
        }
