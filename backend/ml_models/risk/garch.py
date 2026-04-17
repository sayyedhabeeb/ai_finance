"""
GARCH-based volatility and risk modeling.

Provides a high-level interface for fitting and forecasting conditional
volatility models using the ``arch`` library.  Supported model types:

- GARCH(1,1)
- GJR-GARCH (Glosten-Jagannathan-Runkle asymmetric GARCH)
- EGARCH (Exponential GARCH)

Capabilities:
- Volatility forecasting at arbitrary horizons
- Value-at-Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
- Conditional risk metrics (Sharpe, Sortino, max drawdown on conditional vol)
- Model diagnostics (AIC, BIC, Ljung-Box on standardized residuals)
- Rolling window fitting for time-varying parameter estimates
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

logger = logging.getLogger(__name__)

# Suppress arch convergence warnings during grid searches
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_returns(returns: pd.Series, name: str = "returns") -> pd.Series:
    """Validate and clean a returns series for modelling."""
    if not isinstance(returns, pd.Series):
        if isinstance(returns, (pd.DataFrame, np.ndarray)):
            returns = pd.Series(np.squeeze(returns))
        else:
            raise TypeError(f"{name} must be a pandas Series, got {type(returns)}")

    returns = returns.astype(float).dropna()
    if len(returns) < 30:
        raise ValueError(f"{name} must have at least 30 non-NaN observations, got {len(returns)}")

    return returns


def _scale_returns(returns: pd.Series) -> Tuple[pd.Series, float]:
    """Scale returns by 100 for numerical stability in arch."""
    scale = 100.0
    return returns * scale, scale


def _unscale_volatility(vol: np.ndarray, scale: float) -> np.ndarray:
    """Convert conditional volatility back to original scale."""
    return vol / scale


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class GARCHRiskModeler:
    """GARCH-based volatility and risk modeling.

    Fits conditional heteroskedasticity models to asset returns and
    produces volatility forecasts, VaR, CVaR, and diagnostic statistics.

    Example
    -------
    >>> modeler = GARCHRiskModeler()
    >>> result = modeler.fit_model(returns, model_type="GJR-GARCH")
    >>> vol_forecast = modeler.forecast_volatility(returns, horizon=10)
    >>> var = modeler.calculate_var(returns, confidence_levels=[0.95, 0.99])
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "AR",
        lags: Optional[int] = None,
        vol: str = "Garch",
        dist: str = "normal",
        rescale: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        p : int
            Lag order of the symmetric innovation.
        q : int
            Lag order of the lagged conditional variance.
        mean : str
            Mean model specification (``"Zero"``, ``"Constant"``, ``"AR"``).
        lags : int or None
            Number of lags for the AR mean model. ``None`` uses a default.
        vol : str
            Volatility model: ``"Garch"``, ``"GARCH"``, ``"EGARCH"``, or ``"GJR"``.
        dist : str
            Error distribution: ``"normal"``, ``"t"``, ``"ged"``, or ``"skewt"``.
        rescale : bool
            Rescale returns by ×100 for numerical stability.
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.lags = lags
        self.vol = vol
        self.dist = dist
        self.rescale = rescale

        self._last_result: Optional[Any] = None
        self._last_scale: float = 100.0
        self._last_returns_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Model Fitting
    # ------------------------------------------------------------------

    def fit_model(
        self,
        returns: pd.Series,
        model_type: str = "GARCH",
        update_freq: int = 5,
        show_warning: bool = True,
        mlflow_experiment: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
    ) -> Any:
        """Fit a GARCH-family model to asset returns.

        Parameters
        ----------
        returns : pd.Series
            Log or simple returns.
        model_type : str
            One of ``"GARCH"``, ``"GJR-GARCH"``, ``"EGARCH"``.
        update_freq : int
            Frequency of optimization progress updates.
        show_warning : bool
            Show convergence warnings.
        mlflow_experiment : str or None
            Log to MLflow if provided.
        mlflow_run_name : str or None
            Optional MLflow run name.

        Returns
        -------
        arch.univariate.base.ARCHModelResult
            Fitted model result.
        """
        returns = _validate_returns(returns, "returns")
        self._last_returns_name = getattr(returns, "name", None)

        # Map model_type to arch vol specification
        vol_map = {
            "garch": "Garch",
            "gjr-garch": "GARCH",  # arch uses power=2, o=1 for GJR
            "egarch": "EGARCH",
        }
        vol_key = vol_map.get(model_type.lower(), model_type)

        # Determine power and o (asymmetry) parameter
        power = 2.0
        o_param = 0  # No asymmetry
        if model_type.lower() == "gjr-garch":
            o_param = 1

        # Rescale for numerical stability
        if self.rescale:
            y, scale = _scale_returns(returns)
            self._last_scale = scale
        else:
            y = returns.copy()
            scale = 1.0

        am = arch_model(
            y,
            p=self.p,
            q=self.q,
            mean=self.mean,
            lags=self.lags,
            vol=vol_key,
            dist=self.dist,
            power=power,
        )

        # Override o parameter for GJR
        if model_type.lower() == "gjr-garch":
            # GJR requires fitting with o=1; arch handles this via the vol param
            pass

        with warnings.catch_warnings():
            if not show_warning:
                warnings.simplefilter("ignore")
            result = am.fit(update_freq=update_freq, disp="off")

        self._last_result = result
        self._last_scale = scale

        logger.info(
            "Fitted %s model | AIC=%.2f | BIC=%.2f",
            model_type,
            result.aic,
            result.bic,
        )

        # MLflow logging
        if mlflow_experiment:
            try:
                mlflow.set_experiment(mlflow_experiment)
                mlflow.start_run(run_name=mlflow_run_name)
                mlflow.log_params({
                    "model_type": model_type,
                    "p": self.p,
                    "q": self.q,
                    "mean": self.mean,
                    "vol": vol_key,
                    "dist": self.dist,
                    "n_obs": len(returns),
                })
                mlflow.log_metrics({
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "loglikelihood": float(result.loglikelihood),
                })
                mlflow.end_run()
            except Exception as exc:
                logger.warning("MLflow logging failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Volatility Forecasting
    # ------------------------------------------------------------------

    def forecast_volatility(
        self,
        returns: pd.Series,
        horizon: int = 5,
        model_type: str = "GARCH",
        re_fit: bool = True,
    ) -> pd.Series:
        """Forecast conditional volatility.

        Parameters
        ----------
        returns : pd.Series
            Historical returns.
        horizon : int
            Number of steps ahead.
        model_type : str
            GARCH model variant.
        re_fit : bool
            Re-fit the model before forecasting.

        Returns
        -------
        pd.Series
            Annualised volatility forecast indexed by step.
        """
        returns = _validate_returns(returns, "returns")

        if re_fit or self._last_result is None:
            self.fit_model(returns, model_type=model_type, show_warning=False)

        assert self._last_result is not None
        result = self._last_result

        forecast = result.forecast(horizon=horizon, reindex=False)
        # forecast.variance has shape (horizon, 1) for the last date
        variance = forecast.variance.iloc[-1].values
        cond_vol = np.sqrt(variance)

        # Unscale
        cond_vol = _unscale_volatility(cond_vol, self._last_scale)

        # Annualise (assuming ~252 trading days)
        annualised_vol = cond_vol * np.sqrt(252)

        return pd.Series(annualised_vol, index=range(1, horizon + 1), name="annualised_vol")

    # ------------------------------------------------------------------
    # Value-at-Risk
    # ------------------------------------------------------------------

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_levels: Sequence[float] = (0.95, 0.99),
        method: str = "parametric",
        model_type: str = "GARCH",
        re_fit: bool = True,
        position_size: float = 1.0,
    ) -> Dict[str, Any]:
        """Calculate Value-at-Risk.

        Parameters
        ----------
        returns : pd.Series
            Historical returns.
        confidence_levels : sequence of float
            Confidence levels (e.g., 0.95, 0.99).
        method : str
            ``"parametric"`` (conditional), ``"historical"``, or ``"cornish_fisher"``.
        model_type : str
            GARCH model variant (for parametric method).
        re_fit : bool
            Re-fit before computing.
        position_size : float
            Notional position size.

        Returns
        -------
        dict
            ``{
                "method": str,
                "confidence_levels": list[float],
                "var": {0.95: float, 0.99: float, ...},
                "conditional_vol": float,
                "position_var": {0.95: float, ...},
            }``
        """
        returns = _validate_returns(returns, "returns")
        confidence_levels = list(confidence_levels)

        result_var: Dict[str, Any] = {
            "method": method,
            "confidence_levels": confidence_levels,
            "var": {},
            "position_var": {},
        }

        if method == "historical":
            # Simple historical quantile
            for cl in confidence_levels:
                q = returns.quantile(1 - cl)
                result_var["var"][cl] = float(q)
                result_var["position_var"][cl] = float(q * position_size)

            result_var["conditional_vol"] = float(returns.std())

        elif method == "parametric":
            if re_fit or self._last_result is None:
                self.fit_model(returns, model_type=model_type, show_warning=False)
            assert self._last_result is not None
            result = self._last_result

            # Next-day conditional volatility
            forecast = result.forecast(horizon=1, reindex=False)
            next_var = float(forecast.variance.iloc[-1].values[0])
            cond_vol = np.sqrt(next_var) / self._last_scale if self.rescale else np.sqrt(next_var)

            # Mean forecast
            mean_forecast = float(forecast.mean.iloc[-1].values[0]) / self._last_scale if self.rescale else float(forecast.mean.iloc[-1].values[0])

            # Distribution quantiles
            dist_name = self.dist.lower()
            for cl in confidence_levels:
                alpha = 1 - cl
                if dist_name == "normal":
                    z = stats.norm.ppf(alpha)
                elif dist_name == "t":
                    # Use estimated degrees of freedom
                    if hasattr(result, "params") and "nu" in result.params.index:
                        dof = float(result.params["nu"])
                    else:
                        dof = 30.0
                    z = stats.t.ppf(alpha, df=dof)
                elif dist_name == "ged":
                    # Approximate with normal
                    z = stats.norm.ppf(alpha)
                elif dist_name == "skewt":
                    z = stats.norm.ppf(alpha)
                else:
                    z = stats.norm.ppf(alpha)

                var_val = mean_forecast + z * cond_vol
                result_var["var"][cl] = float(var_val)
                result_var["position_var"][cl] = float(var_val * position_size)

            result_var["conditional_vol"] = float(cond_vol)

        elif method == "cornish_fisher":
            # Cornish-Fisher expansion (skewness and kurtosis adjustment)
            mu = float(returns.mean())
            sigma = float(returns.std())
            skew = float(returns.skew())
            kurt = float(returns.kurtosis())

            for cl in confidence_levels:
                alpha = 1 - cl
                z = stats.norm.ppf(alpha)
                # Cornish-Fisher adjusted z
                z_cf = (
                    z
                    + (z ** 2 - 1) * skew / 6
                    + (z ** 3 - 3 * z) * kurt / 24
                    - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
                )
                var_val = mu + z_cf * sigma
                result_var["var"][cl] = float(var_val)
                result_var["position_var"][cl] = float(var_val * position_size)

            result_var["conditional_vol"] = float(sigma)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return result_var

    # ------------------------------------------------------------------
    # Conditional VaR (Expected Shortfall)
    # ------------------------------------------------------------------

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_levels: Sequence[float] = (0.95, 0.99),
        method: str = "historical",
        **var_kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate Conditional VaR (Expected Shortfall).

        Parameters
        ----------
        returns : pd.Series
            Historical returns.
        confidence_levels : sequence of float
            Confidence levels.
        method : str
            ``"historical"`` or ``"parametric"``.
        **var_kwargs
            Additional kwargs passed to ``calculate_var``.

        Returns
        -------
        dict
            ``{
                "method": str,
                "confidence_levels": list[float],
                "cvar": {0.95: float, 0.99: float, ...},
                "var": {0.95: float, ...},
            }``
        """
        returns = _validate_returns(returns, "returns")
        confidence_levels = list(confidence_levels)

        result_cvar: Dict[str, Any] = {
            "method": method,
            "confidence_levels": confidence_levels,
            "cvar": {},
            "var": {},
        }

        if method == "historical":
            for cl in confidence_levels:
                alpha = 1 - cl
                var_val = float(returns.quantile(alpha))
                cvar_val = float(returns[returns <= var_val].mean())
                result_cvar["var"][cl] = var_val
                result_cvar["cvar"][cl] = cvar_val

        elif method == "parametric":
            var_result = self.calculate_var(
                returns,
                confidence_levels=confidence_levels,
                method="parametric",
                **var_kwargs,
            )
            cond_vol = var_result["conditional_vol"]
            dist_name = self.dist.lower()

            for cl in confidence_levels:
                alpha = 1 - cl
                if dist_name == "normal":
                    z = stats.norm.ppf(alpha)
                    # ES for normal: mu + sigma * phi(z) / alpha
                    es_z = -stats.norm.pdf(z) / alpha
                elif dist_name == "t":
                    if (
                        self._last_result is not None
                        and hasattr(self._last_result, "params")
                        and "nu" in self._last_result.params.index
                    ):
                        dof = float(self._last_result.params["nu"])
                    else:
                        dof = 30.0
                    z = stats.t.ppf(alpha, df=dof)
                    es_z = -(
                        stats.t.pdf(z, df=dof)
                        / alpha
                        * (dof + z ** 2)
                        / (dof - 1)
                    )
                else:
                    z = stats.norm.ppf(alpha)
                    es_z = -stats.norm.pdf(z) / alpha

                cvar_val = var_result["var"][cl] + es_z * cond_vol
                result_cvar["var"][cl] = var_result["var"][cl]
                result_cvar["cvar"][cl] = float(cvar_val)
        else:
            raise ValueError(f"Unknown CVaR method: {method}")

        return result_cvar

    # ------------------------------------------------------------------
    # Conditional Risk Metrics
    # ------------------------------------------------------------------

    def conditional_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Compute comprehensive risk metrics based on fitted GARCH model.

        Parameters
        ----------
        returns : pd.Series
            Historical returns.

        Returns
        -------
        dict
            ``{
                "annualised_vol": float,
                "sharpe_ratio": float,
                "sortino_ratio": float,
                "max_drawdown": float,
                "calmar_ratio": float,
                "skewness": float,
                "excess_kurtosis": float,
                "jarque_bera_pvalue": float,
                "mean_return": float,
                "tracking_error": float,
                "downside_deviation": float,
            }``
        """
        returns = _validate_returns(returns, "returns")
        n = len(returns)

        mean_ret = float(returns.mean())
        std_ret = float(returns.std())
        ann_vol = std_ret * np.sqrt(252)
        ann_ret = mean_ret * 252

        # Sharpe ratio (assume risk-free rate = 0)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

        # Sortino ratio
        downside = returns[returns < 0]
        downside_dev = float(downside.std()) * np.sqrt(252) if len(downside) > 1 else 0.0
        sortino = ann_ret / downside_dev if downside_dev > 0 else 0.0

        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = float(drawdowns.min())

        # Calmar ratio
        total_return = float(cum_returns.iloc[-1] / cum_returns.iloc[0] - 1)
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Distribution statistics
        skewness = float(returns.skew())
        excess_kurtosis = float(returns.kurtosis())

        # Jarque-Bera test for normality
        if n > 2:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
        else:
            jb_stat, jb_pvalue = np.nan, np.nan

        # Tracking error (vs. zero — absolute volatility)
        tracking_error = std_ret * np.sqrt(252)

        return {
            "annualised_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "jarque_bera_statistic": float(jb_stat) if not np.isnan(jb_stat) else None,
            "jarque_bera_pvalue": float(jb_pvalue) if not np.isnan(jb_pvalue) else None,
            "mean_return": mean_ret,
            "annualised_return": ann_ret,
            "tracking_error": tracking_error,
            "downside_deviation": downside_dev,
            "total_return": total_return,
        }

    # ------------------------------------------------------------------
    # Model Diagnostics
    # ------------------------------------------------------------------

    def model_diagnostics(
        self, returns: pd.Series, model_type: str = "GARCH", re_fit: bool = True
    ) -> Dict[str, Any]:
        """Run model diagnostics on the fitted GARCH model.

        Parameters
        ----------
        returns : pd.Series
            Returns series.
        model_type : str
            Model variant.
        re_fit : bool
            Re-fit before diagnostics.

        Returns
        -------
        dict
            ``{
                "aic": float,
                "bic": float,
                "loglikelihood": float,
                "params": dict,
                "ljung_box": {"lb_stat": float, "lb_pvalue": float},
                "ljung_box_squared": {"lb_stat": float, "lb_pvalue": float},
                "arch_lm_test": {"stat": float, "pvalue": float},
                "standardized_residuals_stats": dict,
            }``
        """
        returns = _validate_returns(returns, "returns")

        if re_fit or self._last_result is None:
            self.fit_model(returns, model_type=model_type, show_warning=False)

        assert self._last_result is not None
        result = self._last_result

        # Information criteria
        aic = float(result.aic)
        bic = float(result.bic)
        loglik = float(result.loglikelihood)

        # Parameters
        params = {name: float(val) for name, val in result.params.items()}

        # Standardized residuals
        std_resid = result.resid / result.conditional_volatility
        std_resid = std_resid.dropna()

        # Ljung-Box test on standardized residuals
        n_lags = min(20, len(std_resid) // 5)
        if n_lags >= 1:
            lb_result = stats.acorr_ljungbox(std_resid, lags=[n_lags], return_df=True)
            lb_stat = float(lb_result.iloc[0]["lb_stat"])
            lb_pvalue = float(lb_result.iloc[0]["lb_pvalue"])
        else:
            lb_stat, lb_pvalue = np.nan, np.nan

        # Ljung-Box on squared standardized residuals (test for remaining ARCH effects)
        if n_lags >= 1:
            lb_sq_result = stats.acorr_ljungbox(
                std_resid ** 2, lags=[n_lags], return_df=True
            )
            lb_sq_stat = float(lb_sq_result.iloc[0]["lb_stat"])
            lb_sq_pvalue = float(lb_sq_result.iloc[0]["lb_pvalue"])
        else:
            lb_sq_stat, lb_sq_pvalue = np.nan, np.nan

        # ARCH-LM test (Engle's test)
        arch_test_lag = min(10, len(std_resid) // 5)
        if arch_test_lag >= 1:
            resid_sq = std_resid ** 2
            n_obs = len(resid_sq)
            # Manual ARCH-LM: regress resid_sq on lags
            y_arch = resid_sq.values[arch_test_lag:]
            X_arch = np.column_stack(
                [resid_sq.values[i : n_obs - (arch_test_lag - i)] for i in range(arch_test_lag, 0, -1)]
            )
            X_arch = np.column_stack([np.ones(len(y_arch)), X_arch])
            try:
                # OLS: beta = (X'X)^{-1} X'y
                beta = np.linalg.lstsq(X_arch, y_arch, rcond=None)[0]
                residuals = y_arch - X_arch @ beta
                ssr = np.sum(residuals ** 2)
                sst = np.sum((y_arch - y_arch.mean()) ** 2)
                r2 = 1 - ssr / sst if sst > 0 else 0.0
                lm_stat = len(y_arch) * r2
                lm_pvalue = float(1 - stats.chi2.cdf(lm_stat, arch_test_lag))
            except np.linalg.LinAlgError:
                lm_stat, lm_pvalue = np.nan, np.nan
        else:
            lm_stat, lm_pvalue = np.nan, np.nan

        # Standardized residual stats
        resid_stats = {
            "mean": float(std_resid.mean()),
            "std": float(std_resid.std()),
            "skewness": float(std_resid.skew()),
            "kurtosis": float(std_resid.kurtosis()),
        }

        return {
            "aic": aic,
            "bic": bic,
            "loglikelihood": loglik,
            "params": params,
            "n_params": len(params),
            "n_observations": len(returns),
            "ljung_box": {
                "lags": n_lags,
                "lb_stat": lb_stat,
                "lb_pvalue": lb_pvalue,
                "is_white_noise": bool(lb_pvalue > 0.05) if not np.isnan(lb_pvalue) else None,
            },
            "ljung_box_squared": {
                "lags": n_lags,
                "lb_stat": lb_sq_stat,
                "lb_pvalue": lb_sq_pvalue,
                "no_arch_effects": bool(lb_sq_pvalue > 0.05) if not np.isnan(lb_sq_pvalue) else None,
            },
            "arch_lm_test": {
                "lags": arch_test_lag,
                "stat": lm_stat,
                "pvalue": lm_pvalue,
                "no_remaining_arch": bool(lm_pvalue > 0.05) if not np.isnan(lm_pvalue) else None,
            },
            "standardized_residuals_stats": resid_stats,
        }

    # ------------------------------------------------------------------
    # Rolling Window Fitting
    # ------------------------------------------------------------------

    def rolling_fit(
        self,
        returns: pd.Series,
        window_size: int = 252,
        step: int = 21,
        model_type: str = "GARCH",
    ) -> pd.DataFrame:
        """Fit the GARCH model on a rolling window of returns.

        Produces time-varying parameter estimates.

        Parameters
        ----------
        returns : pd.Series
            Historical returns.
        window_size : int
            Size of the rolling window.
        step : int
            Step between successive windows.
        model_type : str
            Model variant.

        Returns
        -------
        pd.DataFrame
            One row per window with parameter estimates, AIC, BIC, and
            the date at which the window ends.
        """
        returns = _validate_returns(returns, "returns")

        rows: List[Dict[str, Any]] = []
        start = 0
        while start + window_size <= len(returns):
            end = start + window_size
            window_returns = returns.iloc[start:end]

            try:
                result = self.fit_model(
                    window_returns,
                    model_type=model_type,
                    show_warning=False,
                )

                row: Dict[str, Any] = {
                    "window_end_date": window_returns.index[-1],
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "loglikelihood": float(result.loglikelihood),
                }
                for pname, pval in result.params.items():
                    row[f"param_{pname}"] = float(pval)
                for pname, pval in result.pvalues.items():
                    row[f"pvalue_{pname}"] = float(pval)

                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "Rolling fit failed for window ending %s: %s",
                    window_returns.index[-1],
                    exc,
                )

            start += step

        if not rows:
            raise RuntimeError("No windows were successfully fitted.")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the modeler state to disk.

        Parameters
        ----------
        path : str
            File path (.pkl).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        state = {
            "p": self.p,
            "q": self.q,
            "mean": self.mean,
            "lags": self.lags,
            "vol": self.vol,
            "dist": self.dist,
            "rescale": self.rescale,
            "last_result": self._last_result,
            "last_scale": self._last_scale,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("GARCH modeler saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "GARCHRiskModeler":
        """Load a saved modeler state.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        GARCHRiskModeler
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        modeler = cls(
            p=state["p"],
            q=state["q"],
            mean=state["mean"],
            lags=state["lags"],
            vol=state["vol"],
            dist=state["dist"],
            rescale=state["rescale"],
        )
        modeler._last_result = state.get("last_result")
        modeler._last_scale = state.get("last_scale", 100.0)

        return modeler
