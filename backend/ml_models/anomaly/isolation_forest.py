"""
Isolation Forest based anomaly detection for financial data.

Provides a comprehensive anomaly detection pipeline that combines
tree-based isolation methods with statistical techniques (z-score, IQR)
for robust detection of anomalies in price, volume, and correlation data.

Features:
- General-purpose anomaly detection on arbitrary feature sets
- Specialised price and volume anomaly detection with feature engineering
- Correlation structure anomaly detection via return matrices
- Ensemble methods combining Isolation Forest with z-score and IQR
- Configurable threshold calibration
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering for Financial Anomaly Detection
# ---------------------------------------------------------------------------

def _engineer_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for price-based anomaly detection.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a ``close`` column.  Optionally ``open``,
        ``high``, ``low``, ``volume``.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame aligned with the input index.
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)

    # Price-based features
    result["close"] = close
    result["return_1d"] = close.pct_change(1)
    result["return_5d"] = close.pct_change(5)
    result["return_10d"] = close.pct_change(10)
    result["return_20d"] = close.pct_change(20)

    # Moving average deviation
    for window in [5, 10, 20, 50]:
        if len(close) >= window:
            sma = close.rolling(window=window, min_periods=1).mean()
            result[f"pct_from_sma_{window}"] = (close - sma) / (sma + 1e-10)

    # Volatility
    result["realized_vol_5d"] = close.pct_change().rolling(5, min_periods=1).std()
    result["realized_vol_20d"] = close.pct_change().rolling(20, min_periods=1).std()

    # Range features
    if all(c in df.columns for c in ("high", "low")):
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        result["intraday_range"] = (high - low) / (close + 1e-10)
        result["range_zscore_20d"] = (
            (result["intraday_range"] - result["intraday_range"].rolling(20, min_periods=1).mean())
            / (result["intraday_range"].rolling(20, min_periods=1).std() + 1e-10)
        )

    # Gap features
    if "open" in df.columns:
        opn = df["open"].astype(float)
        result["overnight_gap"] = (opn - close.shift(1)) / (close.shift(1) + 1e-10)

    # Bollinger Band position
    sma_20 = close.rolling(20, min_periods=1).mean()
    std_20 = close.rolling(20, min_periods=1).std().fillna(0)
    result["bb_position"] = (close - (sma_20 - 2 * std_20)) / (4 * std_20 + 1e-10)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    result["rsi"] = 100 - (100 / (1 + rs))

    return result.fillna(0).replace([np.inf, -np.inf], 0)


def _engineer_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for volume-based anomaly detection.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``volume`` column.

    Returns
    -------
    pd.DataFrame
    """
    result = pd.DataFrame(index=df.index)
    volume = df["volume"].astype(float)

    result["volume"] = volume
    result["volume_1d_change"] = volume.pct_change(1)
    result["volume_5d_change"] = volume.pct_change(5)

    # Volume relative to moving averages
    for window in [5, 10, 20, 50]:
        if len(volume) >= window:
            vol_sma = volume.rolling(window=window, min_periods=1).mean()
            result[f"volume_ratio_sma_{window}"] = volume / (vol_sma + 1e-10)

    # Volume z-score
    result["volume_zscore_20d"] = (
        (volume - volume.rolling(20, min_periods=1).mean())
        / (volume.rolling(20, min_periods=1).std() + 1e-10)
    )

    # Volume-price relationship
    if "close" in df.columns:
        close = df["close"].astype(float)
        price_change = close.pct_change(1).abs()
        result["volume_price_divergence"] = (
            result["volume_zscore_20d"].abs()
            - price_change.rolling(20, min_periods=1).mean().fillna(0) * 100
        )

    return result.fillna(0).replace([np.inf, -np.inf], 0)


# ---------------------------------------------------------------------------
# Statistical Helpers
# ---------------------------------------------------------------------------

def _z_score_anomaly(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Flag anomalies using z-scores.

    Returns a boolean mask where True indicates an anomaly.
    """
    z = np.abs(stats.zscore(values, nan_policy="omit"))
    return z > threshold


def _iqr_anomaly(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Flag anomalies using the IQR method.

    Parameters
    ----------
    values : np.ndarray
    k : float
        IQR multiplier (1.5 = mild, 3.0 = extreme).

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (values < lower) | (values > upper)


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class FinancialAnomalyDetector:
    """Isolation Forest based anomaly detection for financial data.

    Provides specialised methods for price, volume, and correlation
    anomalies.  Combines tree-based methods with statistical techniques
    for robust detection.

    Example
    -------
    >>> detector = FinancialAnomalyDetector(contamination=0.02)
    >>> result = detector.detect_price_anomalies(price_df)
    >>> anomalies = result[result["anomaly_price"] == True]
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Union[str, int, float] = "auto",
        contamination: float = 0.05,
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        z_threshold: float = 3.0,
        iqr_k: float = 1.5,
        mlflow_experiment: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the Isolation Forest.
        max_samples : str, int, or float
            Number of samples to draw per tree.
        contamination : float
            Expected fraction of anomalies (0, 0.5].
        max_features : float
            Fraction of features considered per split.
        bootstrap : bool
            Whether bootstrap samples are used.
        random_state : int
            Random seed.
        n_jobs : int
            Number of parallel jobs (-1 = all cores).
        z_threshold : float
            Z-score threshold for statistical anomaly detection.
        iqr_k : float
            IQR multiplier for statistical anomaly detection.
        mlflow_experiment : str or None
            MLflow experiment name.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.z_threshold = z_threshold
        self.iqr_k = iqr_k
        self.mlflow_experiment = mlflow_experiment

        # Fitted models / scalers (populated during fitting)
        self._price_model: Optional[IsolationForest] = None
        self._price_scaler: Optional[RobustScaler] = None
        self._volume_model: Optional[IsolationForest] = None
        self._volume_scaler: Optional[RobustScaler] = None
        self._general_model: Optional[IsolationForest] = None
        self._general_scaler: Optional[StandardScaler] = None

        # Thresholds (populated during calibration)
        self._price_threshold: float = -0.5
        self._volume_threshold: float = -0.5
        self._general_threshold: float = -0.5

    def _build_isolation_forest(self) -> IsolationForest:
        return IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    # ------------------------------------------------------------------
    # General Anomaly Detection
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Detect anomalies on arbitrary features using Isolation Forest.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        features : list[str] or None
            Columns to use. ``None`` uses all numeric columns.
        fit : bool
            Fit a new model or use an existing one.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added columns:
            ``anomaly`` (bool), ``anomaly_score`` (float, lower = more anomalous).
        """
        result = data.copy()

        if features is None:
            features = data.select_dtypes(include=np.number).columns.tolist()

        if not features:
            raise ValueError("No numeric features available for anomaly detection.")

        X = data[features].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self._general_scaler = StandardScaler()
            X_scaled = self._general_scaler.fit_transform(X)

            self._general_model = self._build_isolation_forest()
            self._general_model.fit(X_scaled)

            # Calibrate threshold: use contamination to derive a score cutoff
            scores = self._general_model.decision_function(X_scaled)
            self._general_threshold = float(np.percentile(scores, self.contamination * 100))

            self._log_model("general", len(features))
        else:
            if self._general_model is None or self._general_scaler is None:
                raise RuntimeError("Model not fitted. Set fit=True or call detect_anomalies(fit=True) first.")
            X_scaled = self._general_scaler.transform(X)

        scores = self._general_model.decision_function(X_scaled)
        predictions = self._general_model.predict(X_scaled)

        result["anomaly"] = predictions == -1
        result["anomaly_score"] = scores

        # Ensemble with z-score and IQR
        z_flags = np.zeros(len(X), dtype=bool)
        iqr_flags = np.zeros(len(X), dtype=bool)
        for i, col in enumerate(features):
            z_flags |= _z_score_anomaly(X[:, i], self.z_threshold)
            iqr_flags |= _iqr_anomaly(X[:, i], self.iqr_k)

        result["anomaly_zscore"] = z_flags
        result["anomaly_iqr"] = iqr_flags
        result["anomaly_ensemble"] = result["anomaly"] | result["anomaly_zscore"] | result["anomaly_iqr"]

        return result

    # ------------------------------------------------------------------
    # Price Anomaly Detection
    # ------------------------------------------------------------------

    def detect_price_anomalies(
        self,
        price_data: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Detect price-related anomalies.

        Parameters
        ----------
        price_data : pd.DataFrame
            Must contain a ``close`` column.  Optional: ``open``, ``high``,
            ``low``, ``volume``.
        fit : bool
            Fit a new model.

        Returns
        -------
        pd.DataFrame
            Input with added ``anomaly_price``, ``price_anomaly_score``, and
            ensemble columns.
        """
        if "close" not in price_data.columns:
            raise ValueError("price_data must contain a 'close' column.")

        result = price_data.copy()

        # Feature engineering
        feat_df = _engineer_price_features(price_data)
        feature_cols = [c for c in feat_df.columns if c != "close"]
        X = feat_df[feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self._price_scaler = RobustScaler()
            X_scaled = self._price_scaler.fit_transform(X)

            self._price_model = self._build_isolation_forest()
            self._price_model.fit(X_scaled)

            scores = self._price_model.decision_function(X_scaled)
            self._price_threshold = float(np.percentile(scores, self.contamination * 100))

            self._log_model("price", len(feature_cols))
        else:
            if self._price_model is None or self._price_scaler is None:
                raise RuntimeError("Price model not fitted.")
            X_scaled = self._price_scaler.transform(X)

        scores = self._price_model.decision_function(X_scaled)
        predictions = self._price_model.predict(X_scaled)

        result["anomaly_price"] = predictions == -1
        result["price_anomaly_score"] = scores

        # Statistical ensemble
        z_flags = np.zeros(len(X), dtype=bool)
        iqr_flags = np.zeros(len(X), dtype=bool)
        for i in range(X.shape[1]):
            z_flags |= _z_score_anomaly(X[:, i], self.z_threshold)
            iqr_flags |= _iqr_anomaly(X[:, i], self.iqr_k)

        result["anomaly_price_zscore"] = z_flags
        result["anomaly_price_iqr"] = iqr_flags
        result["anomaly_price_ensemble"] = (
            result["anomaly_price"] | result["anomaly_price_zscore"] | result["anomaly_price_iqr"]
        )

        return result

    # ------------------------------------------------------------------
    # Volume Anomaly Detection
    # ------------------------------------------------------------------

    def detect_volume_anomalies(
        self,
        price_data: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Detect volume-related anomalies.

        Parameters
        ----------
        price_data : pd.DataFrame
            Must contain a ``volume`` column.
        fit : bool
            Fit a new model.

        Returns
        -------
        pd.DataFrame
            Input with added ``anomaly_volume``, ``volume_anomaly_score``.
        """
        if "volume" not in price_data.columns:
            raise ValueError("price_data must contain a 'volume' column.")

        result = price_data.copy()

        feat_df = _engineer_volume_features(price_data)
        feature_cols = [c for c in feat_df.columns if c != "volume"]
        X = feat_df[feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self._volume_scaler = RobustScaler()
            X_scaled = self._volume_scaler.fit_transform(X)

            self._volume_model = self._build_isolation_forest()
            self._volume_model.fit(X_scaled)

            scores = self._volume_model.decision_function(X_scaled)
            self._volume_threshold = float(np.percentile(scores, self.contamination * 100))

            self._log_model("volume", len(feature_cols))
        else:
            if self._volume_model is None or self._volume_scaler is None:
                raise RuntimeError("Volume model not fitted.")
            X_scaled = self._volume_scaler.transform(X)

        scores = self._volume_model.decision_function(X_scaled)
        predictions = self._volume_model.predict(X_scaled)

        result["anomaly_volume"] = predictions == -1
        result["volume_anomaly_score"] = scores

        # Statistical ensemble
        z_flags = np.zeros(len(X), dtype=bool)
        iqr_flags = np.zeros(len(X), dtype=bool)
        for i in range(X.shape[1]):
            z_flags |= _z_score_anomaly(X[:, i], self.z_threshold)
            iqr_flags |= _iqr_anomaly(X[:, i], self.iqr_k)

        result["anomaly_volume_zscore"] = z_flags
        result["anomaly_volume_iqr"] = iqr_flags
        result["anomaly_volume_ensemble"] = (
            result["anomaly_volume"] | result["anomaly_volume_zscore"] | result["anomaly_volume_iqr"]
        )

        return result

    # ------------------------------------------------------------------
    # Correlation Anomaly Detection
    # ------------------------------------------------------------------

    def detect_correlation_anomalies(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 60,
        z_threshold: float = 2.5,
    ) -> Dict[str, Any]:
        """Detect anomalies in the correlation structure of returns.

        Uses rolling pairwise correlation matrices. Flags periods where
        correlations deviate significantly from historical norms.

        Parameters
        ----------
        returns : pd.DataFrame
            Each column is a different asset's returns.
        lookback_window : int
            Rolling window for computing correlation matrices.
        z_threshold : float
            Z-score threshold for flagging anomalous correlation.

        Returns
        -------
        dict
            ``{
                "correlation_anomaly_dates": list[pd.Timestamp],
                "correlation_zscores": pd.DataFrame,
                "pairwise_max_deviation": dict,
                "mean_correlation_series": pd.Series,
                "anomalous_pairs": list[str],
            }``
        """
        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 return series for correlation analysis.")

        # Compute rolling correlations
        dates: List[Any] = []
        corr_values: List[np.ndarray] = []
        columns = returns.columns.tolist()

        for i in range(lookback_window, len(returns) + 1):
            window = returns.iloc[i - lookback_window : i]
            corr_matrix = window.corr().values
            # Extract upper triangle (pairwise correlations, excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            pair_corrs = corr_matrix[mask]
            dates.append(returns.index[i - 1])
            corr_values.append(pair_corrs)

        # Stack into a matrix: (n_windows, n_pairs)
        corr_matrix_stack = np.column_stack(corr_values) if corr_values else np.array([])

        if corr_matrix_stack.size == 0:
            return {
                "correlation_anomaly_dates": [],
                "correlation_zscores": pd.DataFrame(),
                "pairwise_max_deviation": {},
                "mean_correlation_series": pd.Series(dtype=float),
                "anomalous_pairs": [],
            }

        # Compute z-scores for each pair over time
        mean_corrs = np.nanmean(corr_matrix_stack, axis=1)
        std_corrs = np.nanstd(corr_matrix_stack, axis=1)

        corr_zscores = (corr_matrix_stack - mean_corrs) / (std_corrs + 1e-10)

        # Identify anomalous windows (any pair exceeds threshold)
        max_zscores = np.max(np.abs(corr_zscores), axis=0)
        anomalous_mask = max_zscores > z_threshold

        # Generate pair names
        pair_names: List[str] = []
        idx_upper = np.triu_indices(len(columns), k=1)
        for i, j in zip(idx_upper[0], idx_upper[1]):
            pair_names.append(f"{columns[i]}_{columns[j]}")

        zscore_df = pd.DataFrame(
            corr_zscores.T,
            index=dates,
            columns=pair_names,
        )

        anomalous_dates = [dates[i] for i in range(len(dates)) if anomalous_mask[i]]

        # Find most anomalous pairs
        pairwise_max_dev: Dict[str, float] = {}
        for col_idx, pname in enumerate(pair_names):
            pairwise_max_dev[pname] = float(np.max(np.abs(corr_zscores[col_idx, :])))

        # Sort by max deviation and pick top anomalous pairs
        sorted_pairs = sorted(pairwise_max_dev.items(), key=lambda x: x[1], reverse=True)
        anomalous_pairs = [p for p, s in sorted_pairs if s > z_threshold]

        # Mean correlation series (average of all pairwise correlations per window)
        mean_corr_series = pd.Series(
            np.mean(corr_matrix_stack, axis=0),
            index=dates,
            name="mean_correlation",
        )

        return {
            "correlation_anomaly_dates": anomalous_dates,
            "correlation_zscores": zscore_df,
            "pairwise_max_deviation": pairwise_max_dev,
            "mean_correlation_series": mean_corr_series,
            "anomalous_pairs": anomalous_pairs,
        }

    # ------------------------------------------------------------------
    # Threshold Calibration
    # ------------------------------------------------------------------

    def calibrate_threshold(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        target_false_positive_rate: float = 0.02,
    ) -> float:
        """Calibrate the anomaly detection threshold to achieve a target FPR.

        Parameters
        ----------
        data : pd.DataFrame
            Reference (normal) data for calibration.
        features : list[str] or None
            Feature columns.
        target_false_positive_rate : float
            Desired false positive rate.

        Returns
        -------
        float
            Calibrated threshold (decision function cutoff).
        """
        result = self.detect_anomalies(data, features=features, fit=True)
        scores = result["anomaly_score"].values

        # The threshold should be at the (fpr * 100)-th percentile
        threshold = float(np.percentile(scores, target_false_positive_rate * 100))

        self._general_threshold = threshold
        logger.info(
            "Calibrated general threshold at %.4f for FPR=%.4f",
            threshold,
            target_false_positive_rate,
        )
        return threshold

    # ------------------------------------------------------------------
    # MLflow Logging
    # ------------------------------------------------------------------

    def _log_model(self, model_type: str, n_features: int) -> None:
        """Log model parameters to MLflow."""
        if self.mlflow_experiment:
            try:
                mlflow.set_experiment(self.mlflow_experiment)
                mlflow.start_run(run_name=f"anomaly_{model_type}")
                mlflow.log_params({
                    "model_type": model_type,
                    "n_estimators": self.n_estimators,
                    "contamination": self.contamination,
                    "max_features": self.max_features,
                    "n_features": n_features,
                    "z_threshold": self.z_threshold,
                    "iqr_k": self.iqr_k,
                })
                mlflow.end_run()
            except Exception as exc:
                logger.warning("MLflow logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the detector state to disk.

        Parameters
        ----------
        path : str
            Directory path.
        """
        os.makedirs(path, exist_ok=True)

        state = {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "z_threshold": self.z_threshold,
            "iqr_k": self.iqr_k,
            "max_features": self.max_features,
            "max_samples": self.max_samples,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "price_model": self._price_model,
            "price_scaler": self._price_scaler,
            "volume_model": self._volume_model,
            "volume_scaler": self._volume_scaler,
            "general_model": self._general_model,
            "general_scaler": self._general_scaler,
            "price_threshold": self._price_threshold,
            "volume_threshold": self._volume_threshold,
            "general_threshold": self._general_threshold,
        }
        with open(os.path.join(path, "detector.pkl"), "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Anomaly detector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FinancialAnomalyDetector":
        """Load a saved detector.

        Parameters
        ----------
        path : str
            Directory containing ``detector.pkl``.

        Returns
        -------
        FinancialAnomalyDetector
        """
        with open(os.path.join(path, "detector.pkl"), "rb") as f:
            state = pickle.load(f)

        detector = cls(
            n_estimators=state["n_estimators"],
            contamination=state["contamination"],
            z_threshold=state["z_threshold"],
            iqr_k=state["iqr_k"],
            max_features=state["max_features"],
            max_samples=state["max_samples"],
            bootstrap=state["bootstrap"],
            random_state=state["random_state"],
        )
        detector._price_model = state.get("price_model")
        detector._price_scaler = state.get("price_scaler")
        detector._volume_model = state.get("volume_model")
        detector._volume_scaler = state.get("volume_scaler")
        detector._general_model = state.get("general_model")
        detector._general_scaler = state.get("general_scaler")
        detector._price_threshold = state.get("price_threshold", -0.5)
        detector._volume_threshold = state.get("volume_threshold", -0.5)
        detector._general_threshold = state.get("general_threshold", -0.5)

        return detector
