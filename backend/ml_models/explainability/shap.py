"""
SHAP-based model explainability for financial ML models.

Provides a unified interface for explaining predictions from diverse
model types using SHAP (SHapley Additive exPlanations):

- TreeExplainer for tree-based models (RandomForest, XGBoost, LightGBM)
- KernelExplainer for black-box models (any callable with ``.predict``)
- DeepExplainer support for PyTorch / TensorFlow models
- Feature importance reports
- SHAP value visualization (summary, waterfall, bar plots)
- Portfolio allocation explanation

All plot methods return raw PNG bytes for easy integration with
web APIs or file storage.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Suppress SHAP / matplotlib warnings in headless environments
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ---------------------------------------------------------------------------
# Model Type Detection
# ---------------------------------------------------------------------------

_TREE_LIBRARIES = ("sklearn", "xgboost", "lightgbm", "catboost")
_DEEP_LIBRARIES = ("torch", "tensorflow", "keras")


def _detect_model_type(model: Any) -> str:
    """Detect the type of model to select the appropriate SHAP explainer.

    Returns
    -------
    str
        One of ``"tree"``, ``"deep"``, ``"linear"``, or ``"kernel"``.
    """
    model_module = type(model).__module__.split(".")[0] if hasattr(type(model), "__module__") else ""

    if model_module in _TREE_LIBRARIES:
        return "tree"

    # Check for linear models
    linear_classes = (
        "LinearRegression", "Ridge", "Lasso", "ElasticNet",
        "LogisticRegression", "RidgeClassifier",
    )
    if type(model).__name__ in linear_classes:
        return "linear"

    # Check for deep learning models
    if model_module in _DEEP_LIBRARIES:
        return "deep"

    # Default: kernel explainer (model-agnostic)
    return "kernel"


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class ModelExplainer:
    """SHAP-based model explainability for financial ML models.

    Automatically selects the appropriate SHAP explainer based on model
    type and provides methods for computing and visualising explanations.

    Example
    -------
    >>> explainer = ModelExplainer(model, background_data=X_train)
    >>> explanation = explainer.explain_prediction(X_test.iloc[:5])
    >>> report = explainer.generate_feature_importance_report(X_test)
    >>> png_bytes = explainer.plot_shap_values(explanation, save_path="shap_summary.png")
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[pd.DataFrame] = None,
        model_type: Optional[str] = None,
        nsamples: int = 200,
        link: str = "identity",
        max_evals: int = 500,
    ) -> None:
        """
        Parameters
        ----------
        model : Any
            A fitted ML model (sklearn, xgboost, PyTorch, etc.).
        background_data : pd.DataFrame or None
            Background dataset for SHAP.  Used as the reference
            distribution.  If ``None``, a sample must be provided
            when calling explanation methods.
        model_type : str or None
            Force a specific explainer type (``"tree"``, ``"kernel"``,
            ``"deep"``, ``"linear"``).  ``None`` = auto-detect.
        nsamples : int
            Number of samples for KernelExplainer.
        link : str
            Link function for SHAP values.
        max_evals : int
            Max evaluations for KernelExplainer.
        """
        self.model = model
        self.model_type = model_type or _detect_model_type(model)
        self.nsamples = nsamples
        self.link = link
        self.max_evals = max_evals
        self.background_data = background_data
        self._explainer: Optional[shap.Explainer] = None
        self._feature_names: Optional[List[str]] = None

        # Build explainer if background data is provided
        if background_data is not None:
            self._build_explainer(background_data)

    def _build_explainer(self, background_data: pd.DataFrame) -> shap.Explainer:
        """Build the appropriate SHAP explainer."""
        self.background_data = background_data
        if isinstance(background_data, pd.DataFrame):
            self._feature_names = list(background_data.columns)
            bg_array = background_data.values
        else:
            self._feature_names = None
            bg_array = np.asarray(background_data)

        # Subsample background data for kernel explainer efficiency
        if self.model_type == "kernel" and bg_array.shape[0] > 200:
            rng = np.random.default_rng(42)
            idx = rng.choice(bg_array.shape[0], size=200, replace=False)
            bg_array = bg_array[idx]

        try:
            if self.model_type == "tree":
                self._explainer = shap.TreeExplainer(
                    self.model, data=bg_array, link=self.link
                )

            elif self.model_type == "linear":
                self._explainer = shap.LinearExplainer(
                    self.model, bg_array, link=self.link
                )

            elif self.model_type == "deep":
                self._explainer = shap.DeepExplainer(
                    self.model, bg_array
                )

            else:  # kernel
                def predict_fn(x: np.ndarray) -> np.ndarray:
                    if hasattr(self.model, "predict_proba"):
                        return self.model.predict_proba(x)[:, 1]
                    return self.model.predict(x).flatten()

                self._explainer = shap.KernelExplainer(
                    predict_fn,
                    bg_array,
                    link=self.link,
                    nsamples=self.nsamples,
                )

            logger.info(
                "Built %s SHAP explainer for model type=%s",
                self.model_type,
                type(self.model).__name__,
            )

        except Exception as exc:
            logger.warning(
                "Failed to build %s explainer, falling back to KernelExplainer: %s",
                self.model_type,
                exc,
            )
            self.model_type = "kernel"

            def predict_fn(x: np.ndarray) -> np.ndarray:
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(x)[:, 1]
                return self.model.predict(x).flatten()

            self._explainer = shap.KernelExplainer(
                predict_fn,
                bg_array,
                link=self.link,
                nsamples=self.nsamples,
            )

        return self._explainer

    def _ensure_explainer(self, data: pd.DataFrame) -> shap.Explainer:
        """Ensure the explainer is built."""
        if self._explainer is None:
            return self._build_explainer(data)
        return self._explainer

    # ------------------------------------------------------------------
    # Prediction Explanation
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        input_data: pd.DataFrame,
        background_data: Optional[pd.DataFrame] = None,
    ) -> shap.Explanation:
        """Compute SHAP values for a set of inputs.

        Parameters
        ----------
        input_data : pd.DataFrame
            Instances to explain.
        background_data : pd.DataFrame or None
            Optional background dataset (overrides constructor).

        Returns
        -------
        shap.Explanation
            SHAP explanation object.
        """
        if background_data is not None:
            self._explainer = self._build_explainer(background_data)

        explainer = self._ensure_explainer(input_data)

        if isinstance(input_data, pd.DataFrame):
            feature_names = list(input_data.columns)
            X = input_data.values
        else:
            feature_names = self._feature_names
            X = np.asarray(input_data)

        # Compute SHAP values
        shap_values = explainer.shap_values(X)

        # Handle multi-output SHAP values (e.g., classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class for binary classification

        shap_values = np.array(shap_values)

        # Build SHAP Explanation object
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        # Get base values
        if hasattr(explainer, "expected_value"):
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, tuple, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            base_values = np.full(X.shape[0], float(expected_value))
        else:
            base_values = np.zeros(X.shape[0])

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_values,
            data=X,
            feature_names=feature_names,
        )

        return explanation

    # ------------------------------------------------------------------
    # Feature Importance Report
    # ------------------------------------------------------------------

    def generate_feature_importance_report(
        self,
        X: pd.DataFrame,
        background_data: Optional[pd.DataFrame] = None,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive feature importance report.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset for computing importance.
        background_data : pd.DataFrame or None
            Optional background dataset.
        top_n : int or None
            Return only the top N features.

        Returns
        -------
        dict
            ``{
                "global_importance": pd.DataFrame,
                "mean_abs_shap": dict,
                "feature_ranking": list[str],
                "shap_interaction": dict | None,
                "summary": dict,
            }``
        """
        explanation = self.explain_prediction(X, background_data=background_data)
        shap_values = explanation.values

        if isinstance(explanation, pd.DataFrame):
            feature_names = list(explanation.columns)
        elif hasattr(explanation, "feature_names") and explanation.feature_names is not None:
            feature_names = explanation.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

        # Mean absolute SHAP values per feature (global importance)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Build importance DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": np.mean(shap_values, axis=0),
            "std_shap": np.std(shap_values, axis=0),
            "max_shap": np.max(shap_values, axis=0),
            "min_shap": np.min(shap_values, axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = range(1, len(importance_df) + 1)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        # Feature ranking
        feature_ranking = importance_df["feature"].tolist()

        # Mean SHAP dict
        mean_shap_dict = dict(zip(feature_names, mean_abs_shap))

        # Summary statistics
        total_importance = float(np.sum(mean_abs_shap))
        top_feature = feature_ranking[0] if feature_ranking else None
        top_importance_pct = (
            float(mean_abs_shap[feature_names.index(top_feature)] / total_importance * 100)
            if top_feature and total_importance > 0 else 0.0
        )

        # Cumulative importance
        cum_importance = np.cumsum(np.sort(mean_abs_shap)[::-1])
        n_features_90pct = int(np.searchsorted(cum_importance, 0.90 * total_importance)) + 1

        return {
            "global_importance": importance_df,
            "mean_abs_shap": mean_shap_dict,
            "feature_ranking": feature_ranking,
            "summary": {
                "n_features": len(feature_names),
                "top_feature": top_feature,
                "top_importance_pct": top_importance_pct,
                "n_features_for_90pct": n_features_90pct,
                "total_importance": total_importance,
                "mean_importance_per_feature": float(np.mean(mean_abs_shap)),
                "std_importance": float(np.std(mean_abs_shap)),
            },
        }

    # ------------------------------------------------------------------
    # Portfolio Allocation Explanation
    # ------------------------------------------------------------------

    def explain_portfolio_allocation(
        self,
        weights: Dict[str, float],
        feature_importance: Dict[str, float],
        returns: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Explain portfolio allocation decisions based on feature importance.

        Parameters
        ----------
        weights : dict[str, float]
            Ticker -> portfolio weight.
        feature_importance : dict[str, float]
            Feature -> importance score.
        returns : pd.DataFrame or None
            Historical returns for risk analysis.

        Returns
        -------
        dict
            ``{
                "allocation_contribution": dict,
                "concentration_metrics": dict,
                "risk_attribution": dict,
                "feature_alignment": dict,
            }``
        """
        if not weights:
            return {
                "allocation_contribution": {},
                "concentration_metrics": {"hhi": 0.0, "effective_n": 0.0},
                "risk_attribution": {},
                "feature_alignment": {},
            }

        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])
        total = w.sum()

        # Normalised weights
        w_norm = w / total if total > 0 else w

        # --- Concentration metrics ---
        hhi = float(np.sum(w_norm ** 2))
        effective_n = float(1.0 / hhi) if hhi > 0 else 0.0

        # --- Allocation contribution (assume features may contain ticker prefixes) ---
        allocation_contribution: Dict[str, float] = {}
        for ticker in tickers:
            # Look for features related to this ticker
            matching_importance = 0.0
            n_matching = 0
            for feat, imp in feature_importance.items():
                if ticker.lower() in feat.lower():
                    matching_importance += imp
                    n_matching += 1

            if n_matching > 0:
                avg_imp = matching_importance / n_matching
                allocation_contribution[ticker] = {
                    "weight": float(weights[ticker]),
                    "weight_pct": float(weights[ticker] / total * 100) if total > 0 else 0.0,
                    "avg_feature_importance": float(avg_imp),
                    "n_matching_features": n_matching,
                    "importance_weighted_contribution": float(avg_imp * weights[ticker]),
                }

        # --- Risk attribution ---
        risk_attribution: Dict[str, Any] = {}
        if returns is not None:
            available_tickers = [t for t in tickers if t in returns.columns]
            if available_tickers:
                ticker_returns = returns[available_tickers].dropna()
                vols = ticker_returns.std() * np.sqrt(252)

                for ticker in available_tickers:
                    vol = float(vols.get(ticker, 0.0))
                    w_t = float(weights.get(ticker, 0.0))
                    risk_attribution[ticker] = {
                        "annualised_vol": vol,
                        "portfolio_risk_contribution": w_t * vol,
                        "risk_weight_ratio": (w_t * vol) / (w_norm @ vols.values + 1e-10) if len(available_tickers) > 1 else 1.0,
                    }

        # --- Feature alignment score ---
        if allocation_contribution:
            contributions = [
                v["importance_weighted_contribution"]
                for v in allocation_contribution.values()
            ]
            total_contrib = sum(contributions)
            if total_contrib > 0:
                alignment: Dict[str, float] = {}
                for ticker, v in allocation_contribution.items():
                    alignment[ticker] = v["importance_weighted_contribution"] / total_contrib
            else:
                alignment = {t: 1.0 / len(tickers) for t in tickers}
        else:
            alignment = {t: 1.0 / len(tickers) for t in tickers}

        return {
            "allocation_contribution": allocation_contribution,
            "concentration_metrics": {
                "hhi": hhi,
                "effective_n": effective_n,
                "max_weight": float(np.max(w_norm)),
                "min_weight": float(np.min(w_norm)),
                "n_positions": len(tickers),
            },
            "risk_attribution": risk_attribution,
            "feature_alignment": alignment,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _figure_to_bytes(fig: Figure, dpi: int = 150) -> bytes:
        """Convert a matplotlib Figure to PNG bytes."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    def plot_shap_values(
        self,
        explanation: shap.Explanation,
        plot_type: str = "summary",
        save_path: Optional[str] = None,
        max_display: int = 20,
        dpi: int = 150,
    ) -> bytes:
        """Generate SHAP visualisation plots.

        Parameters
        ----------
        explanation : shap.Explanation
            SHAP explanation object from ``explain_prediction``.
        plot_type : str
            One of ``"summary"``, ``"bar"``, ``"waterfall"``, ``"beeswarm"``, ``"force"``.
        save_path : str or None
            Optional file path to save the PNG.
        max_display : int
            Max features to display.
        dpi : int
            Resolution.

        Returns
        -------
        bytes
            PNG image data.
        """
        plt.close("all")

        if plot_type in ("summary", "beeswarm"):
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(
                explanation.values,
                explanation.data,
                feature_names=explanation.feature_names,
                max_display=max_display,
                show=False,
                plot_type="dot" if plot_type == "summary" else "violin",
            )
            # summary_plot creates its own axes; grab the current figure
            fig = plt.gcf()

        elif plot_type == "bar":
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(
                explanation.values,
                explanation.data,
                feature_names=explanation.feature_names,
                max_display=max_display,
                show=False,
                plot_type="bar",
            )
            fig = plt.gcf()

        elif plot_type == "waterfall":
            # Waterfall plot for the first instance
            fig = plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                explanation[0],
                max_display=max_display,
                show=False,
            )
            fig = plt.gcf()

        elif plot_type == "force":
            fig = plt.figure(figsize=(14, 4))
            shap.force_plot(
                explanation.base_values[0] if hasattr(explanation, "base_values") else 0,
                explanation.values[0],
                explanation.data[0],
                feature_names=explanation.feature_names,
                show=False,
                matplotlib=True,
            )
            fig = plt.gcf()

        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use summary, bar, waterfall, or force.")

        png_bytes = self._figure_to_bytes(fig, dpi=dpi)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(png_bytes)
            logger.info("SHAP plot saved to %s", save_path)

        return png_bytes

    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        background_data: Optional[pd.DataFrame] = None,
        top_n: int = 15,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> bytes:
        """Generate a bar chart of feature importance.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        background_data : pd.DataFrame or None
            Background dataset.
        top_n : int
            Number of top features to display.
        save_path : str or None
            Optional save path.
        dpi : int
            Resolution.

        Returns
        -------
        bytes
            PNG image data.
        """
        report = self.generate_feature_importance_report(X, background_data, top_n=top_n)
        importance_df = report["global_importance"]

        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

        # Plot horizontal bar chart
        y_pos = range(len(importance_df))
        bars = ax.barh(
            y_pos,
            importance_df["mean_abs_shap"].values[::-1],
            color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance_df))),
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df["feature"].values[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("Feature Importance (SHAP)", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels
        for bar, val in zip(bars, importance_df["mean_abs_shap"].values[::-1]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        png_bytes = self._figure_to_bytes(fig, dpi=dpi)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(png_bytes)

        return png_bytes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the explainer state to disk.

        Note: The underlying SHAP explainer may not be fully serialisable.
        Only the configuration is saved; re-build the explainer after loading.

        Parameters
        ----------
        path : str
            Directory path.
        """
        os.makedirs(path, exist_ok=True)

        state = {
            "model_type": self.model_type,
            "nsamples": self.nsamples,
            "link": self.link,
            "max_evals": self.max_evals,
            "feature_names": self._feature_names,
        }

        # Try to pickle the explainer (may fail for some types)
        try:
            with open(os.path.join(path, "explainer.pkl"), "wb") as f:
                pickle.dump(self._explainer, f, protocol=pickle.HIGHEST_PROTOCOL)
            state["explainer_saved"] = True
        except Exception as exc:
            logger.warning("Could not pickle SHAP explainer: %s", exc)
            state["explainer_saved"] = False

        # Save metadata
        with open(os.path.join(path, "explainer_meta.json"), "w") as f:
            import json
            json.dump(state, f, indent=2, default=str)

        logger.info("Model explainer config saved to %s", path)

    @classmethod
    def load(cls, path: str, model: Any) -> "ModelExplainer":
        """Load explainer configuration.

        Parameters
        ----------
        path : str
            Directory path.
        model : Any
            The fitted model object.

        Returns
        -------
        ModelExplainer
        """
        import json

        with open(os.path.join(path, "explainer_meta.json")) as f:
            state = json.load(f)

        explainer = cls(
            model=model,
            model_type=state["model_type"],
            nsamples=state["nsamples"],
            link=state["link"],
            max_evals=state.get("max_evals", 500),
        )
        explainer._feature_names = state.get("feature_names")

        # Try to restore the pickled explainer
        if state.get("explainer_saved"):
            try:
                with open(os.path.join(path, "explainer.pkl"), "rb") as f:
                    explainer._explainer = pickle.load(f)
            except Exception as exc:
                logger.warning("Could not restore SHAP explainer: %s", exc)

        return explainer
