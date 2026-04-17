"""
MLflow Experiment Tracking for AI Financial Brain.

Provides a unified interface for logging experiments, metrics, parameters,
and models across all model types (PatchTST, FinBERT, GARCH, XGBoost, etc.).

Supports both local MLflow and remote MLflow server (e.g., Databricks, self-hosted).
"""

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts")

# Registered model names used by the platform
REGISTERED_MODELS = {
    "patchtst": "PatchTST-Forecaster",
    "finbert": "FinBERT-Sentiment",
    "garch": "GARCH-Volatility",
    "xgboost_regime": "XGBoost-Regime",
    "risk_classifier": "RiskClassifier-Ensemble",
}


class MLflowTracker:
    """
    MLflow experiment tracking wrapper for AI Financial Brain.

    Usage::

        tracker = MLflowTracker()

        with tracker.start_experiment("patchtst_training", "train_v42") as run:
            tracker.log_params({"seq_len": 96, "patch_len": 16, "d_model": 128})
            tracker.log_metrics({"train_loss": 0.034, "val_loss": 0.041, "val_mae": 0.022})
            tracker.log_model(model, artifact_path="model")
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_root: Optional[str] = None,
        experiment_prefix: str = "aifb",
    ):
        self._tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self._artifact_root = artifact_root or ARTIFACT_ROOT
        self._experiment_prefix = experiment_prefix
        self._client = None
        self._active_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy MLflow import & client initialisation
    # ------------------------------------------------------------------

    def _ensure_mlflow(self):
        """Lazy-import mlflow and set tracking URI on first use."""
        if self._client is not None:
            return

        try:
            import mlflow
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_registry_uri(self._tracking_uri)
            self._mlflow = mlflow
            self._client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
            logger.info("MLflow connected to %s", self._tracking_uri)
        except ImportError:
            raise ImportError(
                "mlflow is not installed. Install it with: pip install mlflow"
            )

    def _experiment_name(self, base: str) -> str:
        """Prefix experiment names for organisational clarity."""
        if not base.startswith(self._experiment_prefix):
            return f"{self._experiment_prefix}/{base}"
        return base

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def start_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ):
        """
        Context manager that starts an MLflow run.

        Yields the run info dict so callers can access run_id, etc.

        Usage::

            with tracker.start_experiment("patchtst_training", "train_v42") as run:
                tracker.log_params({...})
                tracker.log_metrics({...})
        """
        self._ensure_mlflow()
        mlflow = self._mlflow

        full_name = self._experiment_name(experiment_name)
        experiment = mlflow.get_experiment_by_name(full_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                full_name,
                artifact_location=f"{self._artifact_root}/{full_name}",
            )
        else:
            experiment_id = experiment.experiment_id

        if description:
            self._client.update_experiment(
                experiment_id,
                description=description,
            )

        default_tags = {
            "platform": "ai_financial_brain",
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if tags:
            default_tags.update(tags)

        return self._RunContext(
            tracker=self,
            experiment_id=experiment_id,
            run_name=run_name,
            tags=default_tags,
        )

    def _set_active_run(self, run_id: str) -> None:
        self._active_run_id = run_id

    def _clear_active_run(self) -> None:
        self._active_run_id = None

    def get_active_run_id(self) -> Optional[str]:
        return self._active_run_id

    # ------------------------------------------------------------------
    # Logging methods (must be called inside start_experiment context)
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters to the active run."""
        self._ensure_mlflow()
        for key, value in params.items():
            self._mlflow.log_param(key, value)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._ensure_mlflow()
        self._mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics to the active run."""
        self._ensure_mlflow()
        self._mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        self._ensure_mlflow()
        self._mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Log a model artifact and optionally register it in the MLflow Model Registry.

        Returns dict with 'model_uri' and optionally 'registered_model_version'.
        """
        self._ensure_mlflow()
        mlflow = self._mlflow

        model_info = mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
        )
        result: Dict[str, str] = {"model_uri": model_info.model_uri}

        if registered_model:
            reg_name = REGISTERED_MODELS.get(registered_model, registered_model)
            try:
                mv = mlflow.register_model(model_info.model_uri, reg_name)
                result["registered_model"] = reg_name
                result["registered_model_version"] = mv.version
                logger.info(
                    "Registered model %s version %s", reg_name, mv.version,
                )
            except Exception as exc:
                logger.warning("Model registration failed: %s", exc)
                result["registration_error"] = str(exc)

        return result

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact."""
        self._ensure_mlflow()
        self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact."""
        self._ensure_mlflow()
        self._mlflow.log_dict(dictionary, artifact_file)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib / plotly figure as an artifact."""
        self._ensure_mlflow()
        self._mlflow.log_figure(figure, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active run."""
        self._ensure_mlflow()
        self._mlflow.set_tag(key, value)

    # ------------------------------------------------------------------
    # Model Registry
    # ------------------------------------------------------------------

    def get_latest_model_version(self, model_name: str) -> Optional[int]:
        """Get the latest version number for a registered model."""
        self._ensure_mlflow()
        reg_name = REGISTERED_MODELS.get(model_name, model_name)
        try:
            versions = self._client.get_latest_versions(reg_name)
            if versions:
                return versions[0].version
        except Exception as exc:
            logger.warning("Could not get latest version for %s: %s", reg_name, exc)
        return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = True,
    ) -> None:
        """
        Transition a registered model version to a new stage
        (None → Staging → Production → Archived).
        """
        self._ensure_mlflow()
        reg_name = REGISTERED_MODELS.get(model_name, model_name)
        self._client.transition_model_version_stage(
            name=reg_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )
        logger.info("Model %s v%s transitioned to %s", reg_name, version, stage)

    def get_model_uri(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """Construct the model URI for loading a registered model."""
        reg_name = REGISTERED_MODELS.get(model_name, model_name)
        return f"models:/{reg_name}/{stage}"

    def compare_runs(
        self,
        experiment_name: str,
        metric: str,
        top_n: int = 5,
        minimize: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search experiment runs sorted by a specific metric.

        Returns list of run summaries.
        """
        self._ensure_mlflow()
        full_name = self._experiment_name(experiment_name)
        order = "ASC" if minimize else "DESC"
        runs = self._mlflow.search_runs(
            experiment_names=[full_name],
            order_by=[f"metrics.{metric} {order}"],
            max_results=top_n,
        )

        results = []
        for _, row in runs.iterrows():
            results.append({
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "metrics": {k.replace("metrics.", ""): v for k, v in row.items() if k.startswith("metrics.") and v is not None},
                "params": {k.replace("params.", ""): v for k, v in row.items() if k.startswith("params.") and v is not None},
            })
        return results

    # ------------------------------------------------------------------
    # Auto-log convenience methods for specific model types
    # ------------------------------------------------------------------

    def autolog_patchtst(
        self,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Convenience method to log a PatchTST training run.

        Args:
            model: The trained PatchTST model instance.
            params: Training parameters (seq_len, patch_len, d_model, n_heads, etc.)
            metrics: Evaluation metrics (train_loss, val_loss, val_mae, val_mse, etc.)
            step: Optional training step / epoch number.
        """
        self._ensure_mlflow()
        self.log_params({
            **params,
            "model_type": "PatchTST",
            "framework": "pytorch",
        })
        self.log_metrics(metrics, step=step)
        return self.log_model(model, artifact_path="patchtst_model", registered_model="patchtst")

    def autolog_finbert(
        self,
        model: Any,
        tokenizer: Any = None,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """
        Convenience method to log a FinBERT fine-tuning run.

        Args:
            model: The fine-tuned FinBERT model.
            tokenizer: The associated tokenizer.
            params: Training parameters (learning_rate, epochs, batch_size, etc.)
            metrics: Evaluation metrics (accuracy, f1, precision, recall, etc.)
        """
        self._ensure_mlflow()
        self.log_params({
            **(params or {}),
            "model_type": "FinBERT",
            "framework": "transformers",
            "base_model": "yiyanghkust/finbert-tone",
        })
        if metrics:
            self.log_metrics(metrics)

        # Save model + tokenizer together
        import mlflow
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmpdir)
            mlflow.log_artifact(tmpdir, artifact_path="finbert_model")

        reg_name = REGISTERED_MODELS["finbert"]
        return {"registered_model": reg_name}

    def autolog_garch(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        forecast_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convenience method to log a GARCH volatility model training run.

        GARCH models are lightweight (arch package) — we log params + metrics
        and serialise the forecast results as a JSON artifact.
        """
        self._ensure_mlflow()
        self.log_params({
            **params,
            "model_type": "GARCH",
            "framework": "arch",
        })
        self.log_metrics(metrics)

        if forecast_dict:
            self.log_dict(forecast_dict, "garch_forecasts.json")

    # ------------------------------------------------------------------
    # Run context manager
    # ------------------------------------------------------------------

    class _RunContext:
        """Context manager wrapping an MLflow active run."""

        def __init__(
            self,
            tracker: "MLflowTracker",
            experiment_id: str,
            run_name: Optional[str],
            tags: Dict[str, str],
        ):
            self._tracker = tracker
            self._experiment_id = experiment_id
            self._run_name = run_name
            self._tags = tags
            self._run = None

        def __enter__(self):
            import mlflow
            self._run = mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=self._run_name,
                tags=self._tags,
            )
            self._tracker._set_active_run(self._run.info.run_id)
            logger.info(
                "MLflow run started: %s (experiment=%s)",
                self._run.info.run_id[:8],
                self._run_name,
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            import mlflow
            if exc_type is not None:
                mlflow.log_param("status", "failed")
                mlflow.set_tag("error", str(exc_val))
                logger.error("MLflow run %s failed: %s", self._run.info.run_id[:8], exc_val)
            mlflow.end_run()
            self._tracker._clear_active_run()
            return False  # don't suppress exceptions

        @property
        def run_id(self) -> str:
            return self._run.info.run_id
