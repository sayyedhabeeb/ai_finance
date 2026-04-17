"""
Automated Model Retraining Pipeline.

Monitors drift detection results and model performance to determine when
retraining is needed. Orchestrates the full retraining lifecycle:
data preparation → training → validation → champion/challenger promotion → A/B test.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REGISTRY_DIR = os.getenv("MODEL_REGISTRY_DIR", "./model_registry")

# Thresholds that trigger retraining
RETRAINING_TRIGGERS = {
    "drift_severity": "high",           # retrain on HIGH or CRITICAL drift
    "performance_drop_pct": 10.0,       # retrain if any key metric drops by 10%
    "days_since_last_retrain": 30,      # retrain at least every 30 days
    "num_predictions": 10000,           # retrain after 10k new predictions logged
    "min_reference_samples": 1000,      # minimum reference data required
}


class RetrainStatus(str, Enum):
    NOT_NEEDED = "not_needed"
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class PromotionResult(str, Enum):
    PROMOTED = "promoted"
    NOT_PROMOTED = "not_promoted"
    AB_TESTING = "ab_testing"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RetrainDecision:
    """Result of checking whether retraining is needed."""
    model_name: str
    should_retrain: bool
    reason: str
    drift_severity: str = "none"
    performance_drop_pct: float = 0.0
    days_since_retrain: float = 0.0
    num_predictions: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RetrainResult:
    """Result of a retraining run."""
    model_name: str
    status: RetrainStatus = RetrainStatus.PENDING
    champion_version: str = ""
    challenger_version: str = ""
    promotion: PromotionResult = PromotionResult.NOT_PROMOTED
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    champion_metrics: Dict[str, float] = field(default_factory=dict)
    improvement: Dict[str, float] = field(default_factory=dict)
    training_time_s: float = 0.0
    started_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    artifact_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status.value,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "promotion": self.promotion.value,
            "training_metrics": {k: round(v, 6) for k, v in self.training_metrics.items()},
            "validation_metrics": {k: round(v, 6) for k, v in self.validation_metrics.items()},
            "champion_metrics": {k: round(v, 6) for k, v in self.champion_metrics.items()},
            "improvement": {k: round(v, 6) for k, v in self.improvement.items()},
            "training_time_s": round(self.training_time_s, 2),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "artifact_path": self.artifact_path,
            "mlflow_run_id": self.mlflow_run_id,
        }


# ---------------------------------------------------------------------------
# Retraining Pipeline
# ---------------------------------------------------------------------------

class RetrainingPipeline:
    """
    Automated model retraining pipeline for AI Financial Brain.

    Workflow:
    1. check_retraining_needed() — evaluate drift + performance metrics
    2. retrain_model() — train a new challenger model
    3. validate_model() — compare challenger vs champion
    4. promote_model() — promote or reject the challenger
    5. (Optional) run_ab_test() — shadow-mode A/B testing

    Usage::

        pipeline = RetrainingPipeline()

        # Check if retraining is needed
        decision = await pipeline.check_retraining_needed("PatchTST", drift_report, perf_report)

        if decision.should_retrain:
            result = await pipeline.retrain_model("PatchTST")
            if result.promotion == PromotionResult.PROMOTED:
                print(f"New model promoted: {result.challenger_version}")
    """

    def __init__(
        self,
        model_registry_dir: str = MODEL_REGISTRY_DIR,
        min_improvement: float = 0.02,  # challenger must be 2% better
        ab_test_traffic: float = 0.1,    # 10% traffic for A/B test
        ab_test_duration_hours: int = 24,
    ):
        self.registry_dir = Path(model_registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.min_improvement = min_improvement
        self.ab_test_traffic = ab_test_traffic
        self.ab_test_duration_hours = ab_test_duration_hours

        # Training function registry — maps model_name -> training callable
        self._trainers: Dict[str, Callable] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Load model metadata from registry."""
        meta_path = self.registry_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._model_metadata = json.load(f)

    def _save_metadata(self) -> None:
        """Persist model metadata to registry."""
        meta_path = self.registry_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self._model_metadata, f, indent=2, default=str)

    def _get_model_meta(self, model_name: str) -> Dict[str, Any]:
        """Get metadata for a specific model."""
        if model_name not in self._model_metadata:
            self._model_metadata[model_name] = {
                "champion_version": "v0.0.0",
                "champion_path": None,
                "last_retrained": None,
                "total_retrains": 0,
                "predictions_since_retrain": 0,
                "ab_test_active": False,
                "ab_test_started": None,
            }
        return self._model_metadata[model_name]

    # ------------------------------------------------------------------
    # Register training functions
    # ------------------------------------------------------------------

    def register_trainer(self, model_name: str, trainer_fn: Callable) -> None:
        """
        Register a training function for a model.

        The trainer function signature must be::

            async def trainer_fn(
                model_name: str,
                artifact_dir: str,
                reference_data: Optional[np.ndarray],
            ) -> Tuple[Any, Dict[str, float]]:
                # ... training logic ...
                return model, {"loss": 0.03, "accuracy": 0.95}
        """
        self._trainers[model_name] = trainer_fn
        logger.info("Registered trainer for model: %s", model_name)

    # ------------------------------------------------------------------
    # Step 1: Check if retraining is needed
    # ------------------------------------------------------------------

    async def check_retraining_needed(
        self,
        model_name: str,
        drift_report: Optional[Any] = None,
        perf_report: Optional[Any] = None,
    ) -> RetrainDecision:
        """
        Determine if a model needs retraining based on:
        - Data drift severity
        - Performance metric degradation
        - Time since last retraining
        - Number of predictions since retraining

        Args:
            model_name: Name of the model to check.
            drift_report: Optional DriftReport from DriftDetector.
            perf_report: Optional PerformanceReport from ModelMonitor.

        Returns:
            RetrainDecision with should_retrain flag and reasoning.
        """
        meta = self._get_model_meta(model_name)
        reasons: List[str] = []

        # 1. Check drift severity
        drift_severity = "none"
        if drift_report is not None:
            if hasattr(drift_report, "overall_severity"):
                drift_severity = drift_report.overall_severity.value
            elif hasattr(drift_report, "dataset_drift"):
                drift_severity = "high" if drift_report.dataset_drift else "none"

        if drift_severity in ("high", "critical"):
            reasons.append(f"drift severity: {drift_severity}")

        # 2. Check performance degradation
        perf_drop = 0.0
        if perf_report is not None and perf_report.metric_changes:
            # Find the worst degradation
            worst_drop = min(perf_report.metric_changes.values())
            perf_drop = abs(worst_drop)
            if perf_drop > RETRAINING_TRIGGERS["performance_drop_pct"] / 100.0:
                reasons.append(
                    f"performance drop: {perf_drop * 100:.1f}%"
                )

        # 3. Check time since last retraining
        days_since = float("inf")
        last_retrained = meta.get("last_retrained")
        if last_retrained:
            last_dt = datetime.fromisoformat(last_retrained)
            days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            if days_since > RETRAINING_TRIGGERS["days_since_last_retrain"]:
                reasons.append(
                    f"days since retrain: {days_since:.0f} > {RETRAINING_TRIGGERS['days_since_last_retrain']}"
                )

        # 4. Check prediction volume
        preds = meta.get("predictions_since_retrain", 0)
        if preds > RETRAINING_TRIGGERS["num_predictions"]:
            reasons.append(
                f"predictions since retrain: {preds} > {RETRAINING_TRIGGERS['num_predictions']}"
            )

        should_retrain = len(reasons) > 0

        decision = RetrainDecision(
            model_name=model_name,
            should_retrain=should_retrain,
            reason="; ".join(reasons) if reasons else "No triggers met",
            drift_severity=drift_severity,
            performance_drop_pct=perf_drop * 100,
            days_since_retrain=days_since if days_since != float("inf") else -1,
            num_predictions=preds,
        )

        logger.info(
            "Retrain check for %s: should=%s, reason=%s",
            model_name,
            should_retrain,
            decision.reason,
        )

        return decision

    # ------------------------------------------------------------------
    # Step 2: Retrain model
    # ------------------------------------------------------------------

    async def retrain_model(
        self,
        model_name: str,
        reference_data: Optional[Any] = None,
        training_params: Optional[Dict[str, Any]] = None,
    ) -> RetrainResult:
        """
        Execute a full retraining run for a model.

        Args:
            model_name: Name of the model to retrain.
            reference_data: Training data (numpy array, dict, or DataFrame).
            training_params: Override default training hyperparameters.

        Returns:
            RetrainResult with full details.
        """
        start = time.time()
        meta = self._get_model_meta(model_name)

        result = RetrainResult(
            model_name=model_name,
            status=RetrainStatus.RUNNING,
            champion_version=meta["champion_version"],
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        # Create artifact directory for this run
        run_id = f"{model_name}_{int(time.time())}"
        artifact_dir = self.registry_dir / "artifacts" / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Check if a trainer is registered
            if model_name in self._trainers:
                trainer_fn = self._trainers[model_name]

                # Prepare reference data as numpy if needed
                ref_np = reference_data
                if reference_data is not None and not isinstance(reference_data, np.ndarray):
                    ref_np = np.array(reference_data)

                model, training_metrics = await trainer_fn(
                    model_name=model_name,
                    artifact_dir=str(artifact_dir),
                    reference_data=ref_np,
                    params=training_params or {},
                )
                result.training_metrics = training_metrics
                result.artifact_path = str(artifact_dir)

                # Save model to artifact directory
                if hasattr(model, "save"):
                    model.save(str(artifact_dir / "model"))

            else:
                # Use built-in placeholder trainer
                logger.warning("No registered trainer for %s — using placeholder", model_name)
                result.training_metrics = self._placeholder_train(model_name, artifact_dir)
                result.artifact_path = str(artifact_dir)

            # Log to MLflow
            try:
                from mlops.tracking.mlflow_tracker import MLflowTracker
                tracker = MLflowTracker()
                with tracker.start_experiment(
                    f"{model_name}_retraining",
                    run_name=run_id,
                    tags={"trigger": "automated_retraining"},
                ):
                    tracker.log_params(training_params or {})
                    tracker.log_metrics(result.training_metrics)
                    result.mlflow_run_id = tracker.get_active_run_id()
            except Exception as exc:
                logger.warning("MLflow logging failed: %s", exc)

            # Increment version
            current_ver = meta["champion_version"]
            major, minor, patch = [int(x) for x in current_ver.lstrip("v").split(".")]
            new_version = f"v{major}.{minor}.{patch + 1}"
            result.challenger_version = new_version

            result.training_time_s = time.time() - start
            result.status = RetrainStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Retraining completed for %s: version=%s, time=%.1fs",
                model_name, new_version, result.training_time_s,
            )

        except Exception as exc:
            result.status = RetrainStatus.FAILED
            result.error = str(exc)
            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.training_time_s = time.time() - start
            logger.exception("Retraining failed for %s", model_name)

        return result

    # ------------------------------------------------------------------
    # Step 3: Validate model (champion vs challenger)
    # ------------------------------------------------------------------

    async def validate_model(
        self,
        model_name: str,
        challenger_result: RetrainResult,
        validation_data: Optional[Tuple[Any, Any]] = None,
    ) -> RetrainResult:
        """
        Validate a challenger model against the current champion.

        Args:
            model_name: Model name.
            challenger_result: The RetrainResult from retrain_model().
            validation_data: Optional (X_val, y_val) for computing metrics.

        Returns:
            Updated RetrainResult with validation_metrics, improvement, and promotion decision.
        """
        meta = self._get_model_meta(model_name)

        # Load champion metrics (from metadata or MLflow)
        champion_metrics = meta.get("champion_metrics", {})

        # If validation data provided, compute metrics on it
        if validation_data is not None:
            X_val, y_val = validation_data
            challenger_metrics = self._evaluate_on_data(model_name, X_val, y_val)
            challenger_result.validation_metrics = challenger_metrics

        challenger_metrics = challenger_result.training_metrics or challenger_result.validation_metrics

        # Compute improvement
        improvement: Dict[str, float] = {}
        for metric_name, challenger_val in challenger_metrics.items():
            champion_val = champion_metrics.get(metric_name)
            if champion_val is not None and champion_val != 0:
                # For metrics where higher is better
                metric_improvement = (challenger_val - champion_val) / abs(champion_val)
                improvement[metric_name] = metric_improvement

        challenger_result.improvement = improvement
        challenger_result.champion_metrics = champion_metrics

        # Determine promotion based on minimum improvement threshold
        avg_improvement = np.mean(list(improvement.values())) if improvement else 0.0

        if avg_improvement >= self.min_improvement:
            challenger_result.promotion = PromotionResult.PROMOTED
        else:
            challenger_result.promotion = PromotionResult.NOT_PROMOTED
            logger.info(
                "Challenger NOT promoted for %s: avg improvement %.4f < threshold %.4f",
                model_name,
                avg_improvement,
                self.min_improvement,
            )

        return challenger_result

    # ------------------------------------------------------------------
    # Step 4: Promote model
    # ------------------------------------------------------------------

    async def promote_model(self, model_name: str, result: RetrainResult) -> RetrainResult:
        """
        Promote the challenger to champion and update the model registry.

        This function:
        1. Backs up the current champion
        2. Points champion to the challenger
        3. Updates metadata
        4. Optionally starts an A/B test
        """
        meta = self._get_model_meta(model_name)
        old_version = meta["champion_version"]
        old_path = meta.get("champion_path")

        # Backup old champion
        if old_path and Path(old_path).exists():
            backup_path = self.registry_dir / "backups" / f"{model_name}_{old_version}"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(old_path, backup_path, dirs_exist_ok=True)
            logger.info("Backed up old champion %s to %s", old_version, backup_path)

        # Promote
        meta["champion_version"] = result.challenger_version
        meta["champion_path"] = result.artifact_path
        meta["champion_metrics"] = result.training_metrics
        meta["last_retrained"] = datetime.now(timezone.utc).isoformat()
        meta["total_retrains"] = meta.get("total_retrains", 0) + 1
        meta["predictions_since_retrain"] = 0

        # Update MLflow registry
        if result.mlflow_run_id:
            try:
                from mlops.tracking.mlflow_tracker import MLflowTracker
                tracker = MLflowTracker()
                tracker.transition_model_stage(model_name, result.challenger_version, "Production")
                if old_version:
                    tracker.transition_model_stage(model_name, old_version, "Archived")
            except Exception as exc:
                logger.warning("MLflow registry update failed: %s", exc)

        self._save_metadata()
        logger.info(
            "Promoted model %s: %s → %s",
            model_name, old_version, result.challenger_version,
        )

        return result

    # ------------------------------------------------------------------
    # Step 5: A/B Testing
    # ------------------------------------------------------------------

    async def run_ab_test(
        self,
        model_name: str,
        challenger_result: RetrainResult,
        traffic_pct: Optional[float] = None,
        duration_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Set up shadow-mode A/B testing for a challenger model.

        The challenger receives `traffic_pct` of requests but responses
        are served from the champion. Challenger metrics are logged for
        comparison after the test duration.
        """
        meta = self._get_model_meta(model_name)
        traffic = traffic_pct or self.ab_test_traffic
        duration = duration_hours or self.ab_test_duration_hours

        # Record A/B test state in metadata
        ab_config = {
            "active": True,
            "challenger_version": challenger_result.challenger_version,
            "champion_version": meta["champion_version"],
            "traffic_pct": traffic,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ends_at": (
                datetime.now(timezone.utc)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .isoformat()
            ),
            "duration_hours": duration,
            "challenger_metrics": {},
            "champion_metrics": {},
        }

        meta["ab_test_active"] = True
        meta["ab_test_config"] = ab_config
        self._save_metadata()

        logger.info(
            "A/B test started for %s: challenger=%s, traffic=%.1f%%, duration=%dh",
            model_name,
            challenger_result.challenger_version,
            traffic * 100,
            duration,
        )

        return ab_config

    def get_ab_test_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the current A/B test configuration and results for a model."""
        meta = self._get_model_meta(model_name)
        config = meta.get("ab_test_config")
        if config and config.get("active"):
            return config
        return None

    def end_ab_test(self, model_name: str, promote_challenger: bool = False) -> Dict[str, Any]:
        """End an A/B test and optionally promote the challenger."""
        meta = self._get_model_meta(model_name)
        config = meta.get("ab_test_config", {})

        meta["ab_test_active"] = False
        meta["ab_test_ended"] = datetime.now(timezone.utc).isoformat()
        self._save_metadata()

        result = {
            "model_name": model_name,
            "challenger_version": config.get("challenger_version"),
            "champion_version": config.get("champion_version"),
            "promoted": promote_challenger,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("A/B test ended for %s: promote=%s", model_name, promote_challenger)
        return result

    # ------------------------------------------------------------------
    # Tracking: Record predictions for retraining triggers
    # ------------------------------------------------------------------

    def record_prediction(self, model_name: str) -> None:
        """Increment the prediction counter for a model."""
        meta = self._get_model_meta(model_name)
        meta["predictions_since_retrain"] = meta.get("predictions_since_retrain", 0) + 1
        self._save_metadata()

    # ------------------------------------------------------------------
    # Full pipeline: check → retrain → validate → promote
    # ------------------------------------------------------------------

    async def run_full_pipeline(
        self,
        model_name: str,
        drift_report: Optional[Any] = None,
        perf_report: Optional[Any] = None,
        reference_data: Optional[Any] = None,
        validation_data: Optional[Tuple[Any, Any]] = None,
    ) -> RetrainResult:
        """
        Run the complete retraining pipeline:
        1. Check if retraining is needed
        2. Retrain the model
        3. Validate the challenger
        4. Promote if validation passes

        Returns the final RetrainResult.
        """
        # Step 1: Check
        decision = await self.check_retraining_needed(
            model_name, drift_report, perf_report
        )

        if not decision.should_retrain:
            logger.info("No retraining needed for %s: %s", model_name, decision.reason)
            return RetrainResult(
                model_name=model_name,
                status=RetrainStatus.NOT_NEEDED,
            )

        # Step 2: Retrain
        result = await self.retrain_model(model_name, reference_data)

        if result.status != RetrainStatus.COMPLETED:
            logger.error("Retraining failed for %s: %s", model_name, result.error)
            return result

        # Step 3: Validate
        result = await self.validate_model(model_name, result, validation_data)

        # Step 4: Promote
        if result.promotion == PromotionResult.PROMOTED:
            result = await self.promote_model(model_name, result)
            logger.info(
                "Full pipeline complete for %s: promoted to %s",
                model_name,
                result.challenger_version,
            )
        else:
            logger.info(
                "Full pipeline complete for %s: challenger not promoted",
                model_name,
            )

        return result

    # ------------------------------------------------------------------
    # Prefect integration
    # ------------------------------------------------------------------

    def create_prefect_flow(self):
        """
        Create a Prefect flow for scheduled retraining checks.

        Checks all registered models and runs retraining where needed.
        """
        try:
            from prefect import flow, get_run_logger
        except ImportError:
            logger.info("Prefect not available for scheduled retraining")
            return None

        pipeline = self

        @flow(name="aifb-retraining-pipeline", log_prints=True)
        def retraining_flow():
            run_logger = get_run_logger()
            run_logger.info("Starting scheduled retraining pipeline")

            for model_name in pipeline._trainers:
                run_logger.info("Checking model: %s", model_name)
                try:
                    # In production, fetch drift/perf reports from the monitoring system
                    decision = pipeline.check_retraining_needed(model_name)
                    if decision.should_retrain:
                        run_logger.info(
                            "Retraining %s: %s", model_name, decision.reason
                        )
                        # Run async in sync Prefect context
                        import asyncio
                        result = asyncio.run(
                            pipeline.run_full_pipeline(model_name)
                        )
                        run_logger.info(
                            "Retraining result for %s: %s",
                            model_name,
                            result.status.value,
                        )
                except Exception as exc:
                    run_logger.error("Pipeline error for %s: %s", model_name, exc)

        return retraining_flow

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _placeholder_train(model_name: str, artifact_dir: Path) -> Dict[str, float]:
        """Generate placeholder training metrics when no trainer is registered."""
        metrics = {
            "train_loss": 0.045,
            "val_loss": 0.052,
            "train_mae": 0.018,
            "val_mae": 0.021,
            "epoch_time_s": 2.3,
        }
        # Write a placeholder artifact
        with open(artifact_dir / "training_config.json", "w") as f:
            json.dump({
                "model_name": model_name,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
        return metrics

    @staticmethod
    def _evaluate_on_data(
        model_name: str,
        X: Any,
        y: Any,
    ) -> Dict[str, float]:
        """Evaluate on provided data — placeholder returns synthetic metrics."""
        return {
            "val_loss": 0.048,
            "val_mae": 0.019,
            "val_mse": 0.0008,
        }
