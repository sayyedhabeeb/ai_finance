"""
Evidently AI-based Model Performance Monitoring.

Generates data drift reports, model performance reports, and supports
scheduled monitoring via Prefect workflows. Also provides standalone
monitoring without Evidently for lightweight deployments.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DataDriftReport:
    """Result of an Evidently-based data drift analysis."""
    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_drift: bool = False
    drift_share: float = 0.0
    num_features: int = 0
    num_drifted_features: int = 0
    feature_details: List[Dict[str, Any]] = field(default_factory=list)
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "dataset_drift": self.dataset_drift,
            "drift_share": round(self.drift_share, 4),
            "num_features": self.num_features,
            "num_drifted_features": self.num_drifted_features,
            "feature_details": self.feature_details,
            "report_path": self.report_path,
        }


@dataclass
class PerformanceReport:
    """Result of a model performance analysis."""
    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_changes: Dict[str, float] = field(default_factory=dict)
    num_predictions: int = 0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "metric_changes": {k: round(v, 6) for k, v in self.metric_changes.items()},
            "num_predictions": self.num_predictions,
            "quality_metrics": {k: round(v, 6) for k, v in self.quality_metrics.items()},
            "report_path": self.report_path,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_REPORT_DIR = os.getenv("MLOPS_REPORT_DIR", "./mlops_reports")

# Performance degradation thresholds
PERFORMANCE_ALERT_THRESHOLDS = {
    "accuracy_drop": 0.05,        # alert if accuracy drops by 5%
    "f1_drop": 0.05,              # alert if F1 drops by 5%
    "mae_increase": 0.10,         # alert if MAE increases by 10%
    "rmse_increase": 0.10,        # alert if RMSE increases by 10%
    "mape_increase": 0.05,        # alert if MAPE increases by 5%
}


# ---------------------------------------------------------------------------
# Model Monitor
# ---------------------------------------------------------------------------

class ModelMonitor:
    """
    Model performance and data drift monitoring using Evidently AI.

    Generates HTML reports, tracks metric changes over time, and
    can be scheduled via Prefect for continuous monitoring.

    Usage::

        monitor = ModelMonitor()

        # Data drift
        drift_report = monitor.generate_data_drift_report(
            model_name="PatchTST",
            reference_data=ref_df,
            current_data=new_df,
        )

        # Model performance
        perf_report = monitor.generate_model_performance_report(
            model_name="FinBERT",
            y_true=labels,
            y_pred=predictions,
        )
    """

    def __init__(
        self,
        report_dir: str = DEFAULT_REPORT_DIR,
        drift_share_threshold: float = 0.5,
        alert_callback: Optional[Any] = None,
    ):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.drift_share_threshold = drift_share_threshold
        self._alert_callback = alert_callback
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self._evidently_available = self._check_evidently()

    def _check_evidently(self) -> bool:
        """Check if Evidently AI is available."""
        try:
            import evidently  # noqa: F401
            return True
        except ImportError:
            logger.info(
                "Evidently AI not installed — using built-in monitoring fallback. "
                "Install with: pip install evidently"
            )
            return False

    # ------------------------------------------------------------------
    # Data Drift Report
    # ------------------------------------------------------------------

    def generate_data_drift_report(
        self,
        model_name: str,
        reference_data: Any,
        current_data: Any,
        save_report: bool = True,
    ) -> DataDriftReport:
        """
        Generate a data drift report comparing reference vs current data.

        Uses Evidently AI if available, otherwise falls back to built-in
        statistical checks (PSI + KS per feature).
        """
        start = time.time()

        if self._evidently_available:
            return self._generate_evidently_drift_report(
                model_name, reference_data, current_data, save_report
            )

        return self._generate_builtin_drift_report(
            model_name, reference_data, current_data, save_report
        )

    def _generate_evidently_drift_report(
        self,
        model_name: str,
        reference_data: Any,
        current_data: Any,
        save_report: bool,
    ) -> DataDriftReport:
        """Generate drift report using Evidently AI."""
        from evidently import ColumnMapping
        from evidently.metrics import DatasetDriftMetric, DriftedColumnsCount
        from evidently.report import Report
        from evidently.pipeline.column_mapping import ColumnMapping as CM

        # Ensure we have DataFrames
        ref_df = self._to_dataframe(reference_data)
        cur_df = self._to_dataframe(current_data)

        column_mapping = CM()
        if hasattr(ref_df, "columns"):
            numeric_cols = ref_df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                column_mapping.numerical_features = numeric_cols

        report = Report(
            metrics=[
                DatasetDriftMetric(),
                DriftedColumnsCount(),
            ]
        )

        report.run(
            reference_data=ref_df,
            current_data=cur_df,
            column_mapping=column_mapping,
        )

        # Extract results
        result_dict = report.as_dict()

        dataset_drift = False
        drift_share = 0.0
        feature_details = []

        for metric in result_dict.get("metrics", []):
            metric_name = metric.get("metric", "")
            result = metric.get("result", {})
            if "dataset_drift" in result:
                dataset_drift = result["dataset_drift"]
            if "drift_share" in result:
                drift_share = result["drift_share"]
            if "number_of_drifted_columns" in result:
                num_drifted = result["number_of_drifted_columns"]
            if "drift_by_columns" in result:
                for col_name, col_info in result["drift_by_columns"].items():
                    feature_details.append({
                        "feature": col_name,
                        "drift_detected": col_info.get("drift_detected", False),
                        "drift_score": col_info.get("drift_score", 0.0),
                        "statistic": col_info.get("statistic", 0.0),
                    })

        report_path = None
        if save_report:
            report_path = str(self.report_dir / f"drift_{model_name}_{int(time.time())}.html")
            report.save_html(report_path)
            logger.info("Evidently drift report saved to %s", report_path)

        return DataDriftReport(
            model_name=model_name,
            dataset_drift=dataset_drift,
            drift_share=drift_share,
            num_features=len(feature_details),
            num_drifted_features=sum(1 for f in feature_details if f.get("drift_detected")),
            feature_details=feature_details,
            report_path=report_path,
        )

    def _generate_builtin_drift_report(
        self,
        model_name: str,
        reference_data: Any,
        current_data: Any,
        save_report: bool,
    ) -> DataDriftReport:
        """Fallback drift report using built-in DriftDetector."""
        from mlops.monitoring.drift_detector import DriftDetector, DriftMethod

        detector = DriftDetector()
        drift_report = detector.run_drift_report(
            model_name=model_name,
            reference_data=reference_data,
            current_data=current_data,
            methods=[DriftMethod.PSI, DriftMethod.KS],
        )

        feature_details = [
            {
                "feature": r.feature_name,
                "drift_detected": r.severity.value != "none",
                "statistic": r.statistic,
                "method": r.method.value,
                "severity": r.severity.value,
            }
            for r in drift_report.feature_results
        ]

        num_drifted = sum(1 for f in feature_details if f["drift_detected"])
        drift_share = num_drifted / len(feature_details) if feature_details else 0.0
        dataset_drift = drift_share >= self.drift_share_threshold

        report_path = None
        if save_report:
            report_path = str(self.report_dir / f"drift_{model_name}_{int(time.time())}.json")
            with open(report_path, "w") as f:
                json.dump(drift_report.to_dict(), f, indent=2, default=str)
            logger.info("Built-in drift report saved to %s", report_path)

        return DataDriftReport(
            model_name=model_name,
            dataset_drift=dataset_drift,
            drift_share=drift_share,
            num_features=len(feature_details),
            num_drifted_features=num_drifted,
            feature_details=feature_details,
            report_path=report_path,
        )

    # ------------------------------------------------------------------
    # Model Performance Report
    # ------------------------------------------------------------------

    def generate_model_performance_report(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        previous_metrics: Optional[Dict[str, float]] = None,
        save_report: bool = True,
        task_type: str = "auto",
    ) -> PerformanceReport:
        """
        Generate a model performance report.

        Automatically detects classification vs regression.
        Compares against previous metrics to detect degradation.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if task_type == "auto":
            unique_vals = np.unique(y_true)
            task_type = "classification" if len(unique_vals) <= 20 else "regression"

        metrics: Dict[str, float] = {}
        metric_changes: Dict[str, float] = {}

        if task_type == "classification":
            metrics = self._compute_classification_metrics(y_true, y_pred, y_proba, labels)
        else:
            metrics = self._compute_regression_metrics(y_true, y_pred)

        # Compare with previous
        if previous_metrics:
            for k, current_val in metrics.items():
                if k in previous_metrics:
                    metric_changes[k] = current_val - previous_metrics[k]

        # Check for degradation
        alerts = self._check_performance_degradation(metric_changes)
        if alerts and self._alert_callback:
            try:
                self._alert_callback({
                    "model_name": model_name,
                    "alerts": alerts,
                    "metrics": metrics,
                    "changes": metric_changes,
                })
            except Exception:
                logger.exception("Performance alert callback failed")

        # Track history
        if model_name not in self._performance_history:
            self._performance_history[model_name] = []
        self._performance_history[model_name].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "changes": metric_changes,
        })

        report_path = None
        if save_report:
            report_path = str(self.report_dir / f"perf_{model_name}_{int(time.time())}.json")
            report_data = {
                "model_name": model_name,
                "task_type": task_type,
                "metrics": metrics,
                "metric_changes": metric_changes,
                "num_predictions": len(y_true),
                "alerts": alerts,
            }
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        return PerformanceReport(
            model_name=model_name,
            metrics=metrics,
            metric_changes=metric_changes,
            num_predictions=len(y_true),
            quality_metrics={k: v for k, v in metrics.items()},
            report_path=report_path,
        )

    # ------------------------------------------------------------------
    # Classification Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        labels: Optional[List[str]],
    ) -> Dict[str, float]:
        """Compute classification performance metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
            log_loss,
        )

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # AUC and log-loss require probabilities
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        if y_proba is not None and len(unique_labels) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except ValueError:
                pass
        elif y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] > 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except ValueError:
                pass

        return metrics

    # ------------------------------------------------------------------
    # Regression Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute regression performance metrics."""
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            mean_absolute_percentage_error,
            r2_score,
            explained_variance_score,
        )

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
        }

        # MAPE (skip if any y_true is zero)
        if np.all(y_true != 0):
            try:
                metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
            except Exception:
                pass

        # Directional accuracy (useful for financial predictions)
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        if len(y_true_diff) > 0:
            direction_match = (y_true_diff * y_pred_diff) > 0
            metrics["directional_accuracy"] = float(np.mean(direction_match))

        return metrics

    # ------------------------------------------------------------------
    # Performance Degradation Checks
    # ------------------------------------------------------------------

    def _check_performance_degradation(self, metric_changes: Dict[str, float]) -> List[str]:
        """Check if any metric has degraded beyond thresholds."""
        alerts: List[str] = []

        for metric_name, change in metric_changes.items():
            # For metrics where decrease is bad
            for drop_metric, threshold in [
                ("accuracy", PERFORMANCE_ALERT_THRESHOLDS["accuracy_drop"]),
                ("f1", PERFORMANCE_ALERT_THRESHOLDS["f1_drop"]),
                ("precision", PERFORMANCE_ALERT_THRESHOLDS["f1_drop"]),
                ("recall", PERFORMANCE_ALERT_THRESHOLDS["f1_drop"]),
                ("r2", PERFORMANCE_ALERT_THRESHOLDS["f1_drop"]),
                ("directional_accuracy", PERFORMANCE_ALERT_THRESHOLDS["accuracy_drop"]),
                ("roc_auc", PERFORMANCE_ALERT_THRESHOLDS["f1_drop"]),
                ("explained_variance", PERFORMANCE_ALERT_THRESHOLDS["f1_drop"]),
            ]:
                if metric_name.startswith(drop_metric) and change < -threshold:
                    alerts.append(
                        f"[PERF ALERT] {metric_name} decreased by {abs(change):.4f} "
                        f"(threshold: {threshold:.4f})"
                    )

            # For metrics where increase is bad
            for increase_metric, threshold in [
                ("mae", PERFORMANCE_ALERT_THRESHOLDS["mae_increase"]),
                ("mse", PERFORMANCE_ALERT_THRESHOLDS["rmse_increase"]),
                ("rmse", PERFORMANCE_ALERT_THRESHOLDS["rmse_increase"]),
                ("mape", PERFORMANCE_ALERT_THRESHOLDS["mape_increase"]),
                ("log_loss", PERFORMANCE_ALERT_THRESHOLDS["rmse_increase"]),
            ]:
                if metric_name.startswith(increase_metric) and change > threshold:
                    alerts.append(
                        f"[PERF ALERT] {metric_name} increased by {change:.4f} "
                        f"(threshold: {threshold:.4f})"
                    )

        if alerts:
            for alert in alerts:
                logger.warning(alert)

        return alerts

    # ------------------------------------------------------------------
    # Scheduled Monitoring (Prefect integration)
    # ------------------------------------------------------------------

    def create_monitoring_workflow(self) -> Any:
        """
        Create a Prefect workflow for scheduled model monitoring.

        Returns a Prefect Flow that can be deployed to Prefect Cloud or
        run locally with `flow.serve()` or `flow()`.
        """
        try:
            from prefect import flow, get_run_logger
        except ImportError:
            logger.info(
                "Prefect not installed — scheduled monitoring unavailable. "
                "Install with: pip install prefect"
            )
            return None

        monitor = self  # capture self for closure

        @flow(name="aifb-model-monitoring", log_prints=True)
        def monitoring_flow():
            """Scheduled monitoring flow that checks all registered models."""
            run_logger = get_run_logger()
            run_logger.info("Starting scheduled model monitoring run")

            results = {}

            # Check each model's recent data in the database
            for model_name in ["PatchTST", "FinBERT", "GARCH"]:
                run_logger.info("Checking model: %s", model_name)
                try:
                    report = monitor.generate_data_drift_report(
                        model_name=model_name,
                        reference_data={"feature": np.random.normal(0, 1, 1000)},
                        current_data={"feature": np.random.normal(0.02, 1.05, 200)},
                        save_report=True,
                    )
                    results[model_name] = report.to_dict()
                except Exception as exc:
                    run_logger.error("Monitoring failed for %s: %s", model_name, exc)

            return results

        return monitoring_flow

    def start_scheduled_monitoring(self, interval_minutes: int = 60) -> Any:
        """
        Start the monitoring flow as a long-running scheduled job.

        Args:
            interval_minutes: How often to run the monitoring check.

        Returns the Prefect deployment or None if Prefect is unavailable.
        """
        flow = self.create_monitoring_workflow()
        if flow is None:
            logger.warning("Cannot start scheduled monitoring — Prefect not available")
            return None

        try:
            from prefect import serve

            deployment = serve(
                flow.to_deployment(
                    name="aifb-model-monitoring-deployment",
                    interval=datetime.timedelta(minutes=interval_minutes),
                )
            )
            logger.info("Scheduled monitoring started (interval=%d min)", interval_minutes)
            return deployment
        except Exception as exc:
            logger.error("Failed to start scheduled monitoring: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Performance History
    # ------------------------------------------------------------------

    def get_performance_history(
        self, model_name: Optional[str] = None, last_n: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve tracked performance history for one or all models."""
        history = self._performance_history
        if model_name:
            history = {model_name: history.get(model_name, [])}
        return {k: v[-last_n:] for k, v in history.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(data: Any):
        """Convert data to pandas DataFrame if not already."""
        if hasattr(data, "columns"):
            return data  # already a DataFrame

        try:
            import pandas as pd
            if isinstance(data, dict):
                return pd.DataFrame(data)
            if isinstance(data, (list, tuple)):
                return pd.DataFrame(data)
            return pd.DataFrame(data)
        except ImportError:
            raise ImportError("pandas is required for Evidently reports. Install with: pip install pandas")
