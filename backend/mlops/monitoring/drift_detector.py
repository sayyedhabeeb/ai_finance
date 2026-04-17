"""
Statistical and ML-based Data / Model Drift Detection.

Provides Population Stability Index (PSI), Kolmogorov-Smirnov test,
Chi-Square test for categorical features, and prediction drift monitoring.
Triggers alerts when drift exceeds configured thresholds.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftMethod(str, Enum):
    PSI = "psi"
    KS = "ks"
    CHI_SQUARE = "chi_square"
    JSDIVERGENCE = "js_divergence"
    WASSERSTEIN = "wasserstein"


# PSI thresholds — industry standard
PSI_THRESHOLDS = {
    DriftSeverity.NONE: 0.1,
    DriftSeverity.LOW: 0.1,
    DriftSeverity.MEDIUM: 0.25,
    DriftSeverity.HIGH: 0.5,
    DriftSeverity.CRITICAL: float("inf"),
}

# KS test p-value thresholds for drift severity
KS_THRESHOLDS = {
    DriftSeverity.NONE: 0.05,
    DriftSeverity.LOW: 0.05,
    DriftSeverity.MEDIUM: 0.01,
    DriftSeverity.HIGH: 0.001,
    DriftSeverity.CRITICAL: 0.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Result of a single drift check."""
    feature_name: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float] = None
    severity: DriftSeverity = DriftSeverity.NONE
    threshold_used: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Aggregate drift report across all features."""
    model_name: str
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    overall_severity: DriftSeverity = DriftSeverity.NONE
    feature_results: List[DriftResult] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "checked_at": self.checked_at,
            "overall_severity": self.overall_severity.value,
            "num_features_checked": len(self.feature_results),
            "num_drifted_features": sum(
                1 for r in self.feature_results if r.severity != DriftSeverity.NONE
            ),
            "results": [
                {
                    "feature": r.feature_name,
                    "method": r.method.value,
                    "statistic": round(r.statistic, 6),
                    "p_value": round(r.p_value, 6) if r.p_value is not None else None,
                    "severity": r.severity.value,
                }
                for r in self.feature_results
            ],
            "alerts": self.alerts,
        }


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Statistical drift detection for ML model inputs and outputs.

    Supports:
    - PSI (Population Stability Index) for numeric features
    - KS test (Kolmogorov-Smirnov) for distribution comparison
    - Wasserstein distance for continuous distributions
    - JS divergence for probability distributions
    - Prediction drift monitoring (output distribution shift)

    Usage::

        detector = DriftDetector()

        # Single feature check
        result = detector.check_psi(reference, current, "feature_a")

        # Full report
        report = detector.run_drift_report(
            model_name="PatchTST",
            reference_data=ref_df,
            current_data=new_df,
        )
    """

    def __init__(
        self,
        psi_threshold: float = 0.1,
        ks_p_threshold: float = 0.05,
        alert_callback: Optional[Any] = None,
    ):
        self.psi_threshold = psi_threshold
        self.ks_p_threshold = ks_p_threshold
        self._alert_callback = alert_callback
        self._drift_history: List[DriftReport] = []

    # ------------------------------------------------------------------
    # PSI (Population Stability Index)
    # ------------------------------------------------------------------

    def check_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        num_bins: int = 10,
        strategy: str = "quantile",
    ) -> DriftResult:
        """
        Calculate Population Stability Index between reference and current distributions.

        PSI formula: Σ (current_pct - reference_pct) × ln(current_pct / reference_pct)

        Args:
            reference: Baseline distribution array.
            current: New distribution array.
            feature_name: Name of the feature for reporting.
            num_bins: Number of bins for the histogram.
            strategy: Binning strategy — 'quantile' or 'uniform'.

        Returns:
            DriftResult with PSI statistic and severity classification.
        """
        ref = np.array(reference, dtype=np.float64)
        cur = np.array(current, dtype=np.float64)

        # Handle edge cases
        if len(ref) < 10 or len(cur) < 10:
            return DriftResult(
                feature_name=feature_name,
                method=DriftMethod.PSI,
                statistic=0.0,
                severity=DriftSeverity.NONE,
                details={"error": "insufficient data points"},
            )

        # Compute bin edges
        combined = np.concatenate([ref, cur])
        if strategy == "quantile":
            percentiles = np.linspace(0, 100, num_bins + 1)
            edges = np.unique(np.nanpercentile(combined, percentiles))
        else:
            edges = np.linspace(np.nanmin(combined), np.nanmax(combined), num_bins + 1)

        if len(edges) < 3:
            edges = np.linspace(np.nanmin(combined), np.nanmax(combined), num_bins + 2)

        # Compute histograms
        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)

        # Normalise to percentages
        ref_pct = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else ref_counts
        cur_pct = cur_counts / cur_counts.sum() if cur_counts.sum() > 0 else cur_counts

        # Add small epsilon to avoid log(0)
        eps = 1e-6
        ref_pct = ref_pct + eps
        cur_pct = cur_pct + eps

        # PSI formula
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        # Classify severity
        if psi < PSI_THRESHOLDS[DriftSeverity.LOW]:
            severity = DriftSeverity.NONE
        elif psi < PSI_THRESHOLDS[DriftSeverity.MEDIUM]:
            severity = DriftSeverity.LOW
        elif psi < PSI_THRESHOLDS[DriftSeverity.HIGH]:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        return DriftResult(
            feature_name=feature_name,
            method=DriftMethod.PSI,
            statistic=psi,
            severity=severity,
            threshold_used=self.psi_threshold,
            details={
                "num_bins": len(edges) - 1,
                "strategy": strategy,
                "ref_samples": len(ref),
                "cur_samples": len(cur),
            },
        )

    # ------------------------------------------------------------------
    # KS Test
    # ------------------------------------------------------------------

    def check_ks(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
    ) -> DriftResult:
        """
        Kolmogorov-Smirnov two-sample test for distribution equality.

        Returns:
            DriftResult with KS statistic, p-value, and severity.
        """
        ref = np.array(reference, dtype=np.float64)
        cur = np.array(current, dtype=np.float64)

        if len(ref) < 5 or len(cur) < 5:
            return DriftResult(
                feature_name=feature_name,
                method=DriftMethod.KS,
                statistic=0.0,
                p_value=1.0,
                severity=DriftSeverity.NONE,
                details={"error": "insufficient data points for KS test"},
            )

        from scipy.stats import ks_2samp

        ks_stat, p_value = ks_2samp(ref, cur)

        # Low p-value indicates drift
        if p_value > KS_THRESHOLDS[DriftSeverity.LOW]:
            severity = DriftSeverity.NONE
        elif p_value > KS_THRESHOLDS[DriftSeverity.MEDIUM]:
            severity = DriftSeverity.LOW
        elif p_value > KS_THRESHOLDS[DriftSeverity.HIGH]:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        return DriftResult(
            feature_name=feature_name,
            method=DriftMethod.KS,
            statistic=float(ks_stat),
            p_value=float(p_value),
            severity=severity,
            threshold_used=self.ks_p_threshold,
            details={
                "ref_samples": len(ref),
                "cur_samples": len(cur),
            },
        )

    # ------------------------------------------------------------------
    # Wasserstein Distance
    # ------------------------------------------------------------------

    def check_wasserstein(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
    ) -> DriftResult:
        """
        Earth Mover's (Wasserstein-1) distance between distributions.
        More sensitive to shifts in the center of the distribution than KS.
        """
        ref = np.array(reference, dtype=np.float64)
        cur = np.array(current, dtype=np.float64)

        from scipy.stats import wasserstein_distance

        distance = float(wasserstein_distance(ref, cur))

        # Normalise by reference std for threshold
        ref_std = float(np.std(ref))
        normalised = distance / ref_std if ref_std > 0 else 0.0

        if normalised < 0.1:
            severity = DriftSeverity.NONE
        elif normalised < 0.3:
            severity = DriftSeverity.LOW
        elif normalised < 0.5:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        return DriftResult(
            feature_name=feature_name,
            method=DriftMethod.WASSERSTEIN,
            statistic=distance,
            severity=severity,
            details={
                "normalised_distance": round(normalised, 4),
                "ref_std": round(ref_std, 6),
            },
        )

    # ------------------------------------------------------------------
    # Jensen-Shannon Divergence
    # ------------------------------------------------------------------

    def check_js_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        num_bins: int = 20,
    ) -> DriftResult:
        """
        Jensen-Shannon divergence — symmetric version of KL divergence.
        Returns values in [0, 1] for discrete distributions.
        """
        ref = np.array(reference, dtype=np.float64)
        cur = np.array(current, dtype=np.float64)

        combined = np.concatenate([ref, cur])
        edges = np.linspace(np.nanmin(combined), np.nanmax(combined), num_bins + 1)

        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)

        ref_pct = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else ref_counts + 1e-10
        cur_pct = cur_counts / cur_counts.sum() if cur_counts.sum() > 0 else cur_counts + 1e-10

        from scipy.spatial.distance import jensenshannon
        jsd = float(jensenshannon(ref_pct, cur_pct, base=2))

        # JSD is bounded [0, 1] for base-2
        if jsd < 0.02:
            severity = DriftSeverity.NONE
        elif jsd < 0.05:
            severity = DriftSeverity.LOW
        elif jsd < 0.1:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        return DriftResult(
            feature_name=feature_name,
            method=DriftMethod.JSDIVERGENCE,
            statistic=jsd,
            severity=severity,
            details={"num_bins": num_bins},
        )

    # ------------------------------------------------------------------
    # Prediction Drift
    # ------------------------------------------------------------------

    def check_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        model_name: str = "unknown",
    ) -> DriftResult:
        """
        Check for drift in model output distribution.
        Combines PSI and KS for a robust assessment.
        """
        ref = np.array(reference_predictions, dtype=np.float64)
        cur = np.array(current_predictions, dtype=np.float64)

        psi_result = self.check_psi(ref, cur, f"{model_name}_pred_psi")
        ks_result = self.check_ks(ref, cur, f"{model_name}_pred_ks")

        # Use the more severe of the two
        severities = [DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        max_sev = max(
            [severities.index(psi_result.severity), severities.index(ks_result.severity)]
        )

        return DriftResult(
            feature_name=f"{model_name}_predictions",
            method=DriftMethod.PSI,
            statistic=max(psi_result.statistic, ks_result.statistic),
            p_value=ks_result.p_value,
            severity=severities[max_sev],
            details={
                "psi": psi_result.statistic,
                "ks_stat": ks_result.statistic,
                "ks_pvalue": ks_result.p_value,
                "ref_mean": float(np.mean(ref)),
                "cur_mean": float(np.mean(cur)),
                "mean_shift_pct": float(abs(np.mean(cur) - np.mean(ref)) / (abs(np.mean(ref)) + 1e-10) * 100),
            },
        )

    # ------------------------------------------------------------------
    # Full Drift Report
    # ------------------------------------------------------------------

    def run_drift_report(
        self,
        model_name: str,
        reference_data: Any,
        current_data: Any,
        numeric_features: Optional[List[str]] = None,
        prediction_column: Optional[str] = None,
        methods: Optional[List[DriftMethod]] = None,
    ) -> DriftReport:
        """
        Run a comprehensive drift check across all specified features.

        Args:
            model_name: Name of the model being monitored.
            reference_data: Reference/baseline data (dict, list of dicts, or DataFrame-like).
            current_data: Current/new data (same format as reference).
            numeric_features: List of numeric feature column names to check.
            prediction_column: Name of the prediction column for output drift.
            methods: Drift methods to use (default: PSI + KS).

        Returns:
            DriftReport with per-feature results and overall severity.
        """
        start = time.time()
        methods = methods or [DriftMethod.PSI, DriftMethod.KS]

        # Convert to dict-of-arrays format
        ref_dict = self._to_dict_of_arrays(reference_data)
        cur_dict = self._to_dict_of_arrays(current_data)

        # Auto-detect features if not specified
        if numeric_features is None:
            numeric_features = list(ref_dict.keys())
            # Filter to truly numeric
            numeric_features = [
                k for k in numeric_features
                if isinstance(ref_dict[k], (np.ndarray, list))
                and len(ref_dict[k]) > 0
                and isinstance(ref_dict[k][0], (int, float, np.integer, np.floating))
            ]

        report = DriftReport(
            model_name=model_name,
            metadata={
                "ref_samples": len(next(iter(ref_dict.values()))),
                "cur_samples": len(next(iter(cur_dict.values()))),
                "features_checked": len(numeric_features),
                "methods": [m.value for m in methods],
                "computation_time_s": 0.0,
            },
        )

        max_severity_idx = 0
        severities = [DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]

        for feat in numeric_features:
            ref_arr = np.array(ref_dict.get(feat, []), dtype=np.float64)
            cur_arr = np.array(cur_dict.get(feat, []), dtype=np.float64)

            if len(ref_arr) == 0 or len(cur_arr) == 0:
                continue

            for method in methods:
                match method:
                    case DriftMethod.PSI:
                        result = self.check_psi(ref_arr, cur_arr, feat)
                    case DriftMethod.KS:
                        result = self.check_ks(ref_arr, cur_arr, feat)
                    case DriftMethod.WASSERSTEIN:
                        result = self.check_wasserstein(ref_arr, cur_arr, feat)
                    case DriftMethod.JSDIVERGENCE:
                        result = self.check_js_divergence(ref_arr, cur_arr, feat)
                    case _:
                        continue

                report.feature_results.append(result)

                sev_idx = severities.index(result.severity)
                if sev_idx > max_severity_idx:
                    max_severity_idx = sev_idx

                if result.severity.value in (DriftSeverity.HIGH.value, DriftSeverity.CRITICAL.value):
                    alert_msg = (
                        f"[DRIFT ALERT] {model_name} | Feature: {feat} | "
                        f"Method: {method.value} | Severity: {result.severity.value} | "
                        f"Statistic: {result.statistic:.6f}"
                    )
                    report.alerts.append(alert_msg)
                    logger.warning(alert_msg)

        report.overall_severity = severities[max_severity_idx]
        report.metadata["computation_time_s"] = round(time.time() - start, 3)

        # Trigger callback if configured
        if report.alerts and self._alert_callback is not None:
            try:
                self._alert_callback(report)
            except Exception:
                logger.exception("Alert callback failed")

        self._drift_history.append(report)
        logger.info(
            "Drift report for %s: severity=%s, features=%d, alerts=%d, time=%.3fs",
            model_name,
            report.overall_severity.value,
            len(report.feature_results),
            len(report.alerts),
            report.metadata["computation_time_s"],
        )

        return report

    def get_drift_history(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return stored drift reports, optionally filtered by model name."""
        reports = self._drift_history
        if model_name:
            reports = [r for r in reports if r.model_name == model_name]
        return [r.to_dict() for r in reports]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict_of_arrays(data: Any) -> Dict[str, np.ndarray]:
        """
        Convert various data formats to a dict of numpy arrays.
        Accepts: dict of lists, dict of arrays, list of dicts, or pandas DataFrame.
        """
        if hasattr(data, "columns"):
            # pandas DataFrame
            return {col: data[col].values for col in data.columns}

        if isinstance(data, dict):
            return {k: np.array(v) for k, v in data.items()}

        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
            # list of dicts
            keys = data[0].keys()
            return {k: np.array([d.get(k) for d in data]) for k in keys}

        raise ValueError(f"Unsupported data format: {type(data)}. Expected dict, list[dict], or DataFrame.")
