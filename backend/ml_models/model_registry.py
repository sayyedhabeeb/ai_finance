"""
Central registry for managing all ML models.

Provides a singleton-based model registry with:
- Model registration with metadata, metrics, and parameters
- Versioned model loading (latest or specific version)
- Pre-warmed model cache for fast inference
- MLflow model registry integration under the hood
- Model lifecycle management (staging, archiving)

Example
-------
>>> registry = ModelRegistry.get_instance(tracking_uri="sqlite:///mlflow.db")
>>> registry.register_model("patchtst_v1", model, metrics={"mse": 0.001}, params={"lr": 1e-4})
>>> model = registry.load_model("patchtst_v1")
>>> info = registry.get_model_info("patchtst_v1")
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ModelInfo:
    """Metadata container for a registered model.

    Attributes
    ----------
    model_name : str
        Unique model identifier.
    version : str
        Model version string.
    metrics : dict
        Performance metrics.
    params : dict
        Hyperparameters / configuration.
    status : str
        One of ``"registered"``, ``"staging"``, ``"production"``, ``"archived"``.
    created_at : str
        ISO 8601 timestamp.
    updated_at : str
        ISO 8601 timestamp.
    description : str or None
        Human-readable description.
    tags : dict
        Arbitrary tags for filtering.
    """

    def __init__(
        self,
        model_name: str,
        version: str,
        metrics: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        status: str = "registered",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.metrics = metrics or {}
        self.params = params or {}
        self.status = status
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.description = description
        self.tags = tags or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "metrics": self.metrics,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        info = cls(
            model_name=data["model_name"],
            version=data["version"],
            metrics=data.get("metrics", {}),
            params=data.get("params", {}),
            status=data.get("status", "registered"),
            description=data.get("description"),
            tags=data.get("tags", {}),
        )
        info.created_at = data.get("created_at", info.created_at)
        info.updated_at = data.get("updated_at", info.updated_at)
        return info

    def __repr__(self) -> str:
        return (
            f"ModelInfo(name={self.model_name!r}, version={self.version!r}, "
            f"status={self.status!r})"
        )


class ModelRegistry:
    """Central registry for managing all ML models.

    Uses a singleton pattern so that all components share the same
    registry instance.  Integrates with MLflow model registry when
    available.

    Parameters
    ----------
    storage_path : str
        Directory to persist registry metadata.
    tracking_uri : str or None
        MLflow tracking URI.  ``None`` disables MLflow integration.
    cache_size : int
        Max number of pre-warmed models in memory.
    prewarm_models : list[str] or None
        Model names to pre-warm on initialisation.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        storage_path: str = "./model_registry",
        tracking_uri: Optional[str] = None,
        cache_size: int = 10,
        prewarm_models: Optional[List[str]] = None,
    ) -> None:
        self.storage_path = storage_path
        self.tracking_uri = tracking_uri
        self.cache_size = cache_size

        # In-memory stores
        self._models: Dict[str, Dict[str, ModelInfo]] = {}  # name -> {version: info}
        self._cache: Dict[str, Any] = {}  # "name/version" -> model object
        self._cache_access_times: Dict[str, float] = {}  # For LRU eviction
        self._mutex = threading.RLock()

        # MLflow integration
        self._mlflow_client: Optional[Any] = None
        self._mlflow_available = False
        if tracking_uri:
            self._init_mlflow(tracking_uri)

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        # Load persisted registry
        self._load_registry()

        # Pre-warm models
        if prewarm_models:
            for name in prewarm_models:
                try:
                    self.load_model(name)
                except Exception as exc:
                    logger.warning("Failed to pre-warm model %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        storage_path: str = "./model_registry",
        tracking_uri: Optional[str] = None,
        cache_size: int = 10,
        prewarm_models: Optional[List[str]] = None,
    ) -> "ModelRegistry":
        """Get the singleton instance of the ModelRegistry.

        Parameters are only used on the first call; subsequent calls
        return the existing instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        storage_path=storage_path,
                        tracking_uri=tracking_uri,
                        cache_size=cache_size,
                        prewarm_models=prewarm_models,
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # MLflow Integration
    # ------------------------------------------------------------------

    def _init_mlflow(self, tracking_uri: str) -> None:
        """Initialise MLflow client."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(tracking_uri)
            self._mlflow_client = MlflowClient(tracking_uri=tracking_uri)
            self._mlflow_available = True
            logger.info("MLflow integration enabled (uri=%s)", tracking_uri)
        except ImportError:
            logger.warning("MLflow not installed. Falling back to local registry.")
        except Exception as exc:
            logger.warning("MLflow initialization failed: %s", exc)

    # ------------------------------------------------------------------
    # Model Registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        model: Any,
        metrics: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        status: str = "registered",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a model in the registry.

        Parameters
        ----------
        model_name : str
            Unique identifier for the model.
        model : Any
            The model object (must be pickle-serialisable).
        metrics : dict or None
            Performance metrics (e.g., ``{"mse": 0.001, "r2": 0.95}``).
        params : dict or None
            Hyperparameters.
        version : str or None
            Version string. ``None`` auto-increments.
        status : str
            Model status.
        description : str or None
            Description.
        tags : dict or None
            Tags for filtering.

        Returns
        -------
        str
            The assigned version string.
        """
        with self._mutex:
            # Determine version
            if version is None:
                existing = self._models.get(model_name, {})
                next_num = max(
                    (int(v.split("v")[1]) for v in existing if v.startswith("v")),
                    default=0,
                ) + 1
                version = f"v{next_num}"
            else:
                existing = self._models.get(model_name, {})
                if version in existing:
                    logger.warning(
                        "Overwriting existing model %s version %s",
                        model_name,
                        version,
                    )

            # Create ModelInfo
            info = ModelInfo(
                model_name=model_name,
                version=version,
                metrics=metrics or {},
                params=params or {},
                status=status,
                description=description,
                tags=tags or {},
            )

            # Store metadata
            if model_name not in self._models:
                self._models[model_name] = {}
            self._models[model_name][version] = info

            # Store model in cache
            cache_key = f"{model_name}/{version}"
            self._cache[cache_key] = model
            self._cache_access_times[cache_key] = time.time()
            self._evict_cache()

            # Persist to disk
            self._save_registry()
            self._save_model(model_name, version, model)

            # MLflow logging
            if self._mlflow_available:
                self._log_to_mlflow(model_name, version, model, info)

            logger.info(
                "Registered model %s version %s (status=%s)",
                model_name,
                version,
                status,
            )
            return version

    def _log_to_mlflow(
        self, model_name: str, version: str, model: Any, info: ModelInfo
    ) -> None:
        """Log model metadata to MLflow."""
        try:
            import mlflow

            experiment_name = f"registry_{model_name}"
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"{model_name}_{version}"):
                mlflow.log_params({
                    "model_name": model_name,
                    "version": version,
                    "status": info.status,
                    **info.params,
                })
                mlflow.log_metrics({
                    k: float(v) for k, v in info.metrics.items() if isinstance(v, (int, float))
                })
                if info.tags:
                    for tk, tv in info.tags.items():
                        mlflow.set_tag(tk, tv)
                if info.description:
                    mlflow.set_tag("description", info.description)
        except Exception as exc:
            logger.warning("MLflow logging failed for %s/%s: %s", model_name, version, exc)

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Any:
        """Load a model from the registry.

        If ``version`` is ``None``, loads the latest version.

        Parameters
        ----------
        model_name : str
            Model identifier.
        version : str or None
            Specific version, or ``None`` for latest.

        Returns
        -------
        Any
            The loaded model object.
        """
        with self._mutex:
            versions = self._models.get(model_name)
            if not versions:
                raise KeyError(f"Model '{model_name}' not found in registry.")

            if version is None:
                # Get latest version (highest v-number)
                version_numbers = []
                for v in versions:
                    if v.startswith("v"):
                        try:
                            version_numbers.append((int(v[1:]), v))
                        except ValueError:
                            pass
                if not version_numbers:
                    version = list(versions.keys())[0]
                else:
                    version_numbers.sort(reverse=True)
                    version = version_numbers[0][1]

            if version not in versions:
                raise KeyError(
                    f"Version '{version}' not found for model '{model_name}'. "
                    f"Available: {list(versions.keys())}"
                )

            # Check cache first
            cache_key = f"{model_name}/{version}"
            if cache_key in self._cache:
                self._cache_access_times[cache_key] = time.time()
                return self._cache[cache_key]

            # Load from disk
            model = self._load_model_from_disk(model_name, version)
            if model is None:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' version '{version}' from disk."
                )

            # Cache it
            self._cache[cache_key] = model
            self._cache_access_times[cache_key] = time.time()
            self._evict_cache()

            logger.info("Loaded model %s version %s from disk", model_name, version)
            return model

    def _load_model_from_disk(self, model_name: str, version: str) -> Any:
        """Load model pickle from disk."""
        path = os.path.join(self.storage_path, "models", model_name, version, "model.pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", path, exc)
            return None

    def _save_model(self, model_name: str, version: str, model: Any) -> None:
        """Persist model to disk."""
        path = os.path.join(self.storage_path, "models", model_name, version)
        os.makedirs(path, exist_ok=True)
        try:
            with open(os.path.join(path, "model.pkl"), "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.error("Failed to save model to %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Cache Management
    # ------------------------------------------------------------------

    def _evict_cache(self) -> None:
        """Evict least-recently-used models when cache is full."""
        while len(self._cache) > self.cache_size:
            oldest_key = min(self._cache_access_times, key=self._cache_access_times.get)
            self._cache.pop(oldest_key, None)
            self._cache_access_times.pop(oldest_key, None)
            logger.debug("Evicted model %s from cache", oldest_key)

    def warm_cache(self, model_names: List[Tuple[str, Optional[str]]]) -> None:
        """Pre-warm the cache by loading specified models.

        Parameters
        ----------
        model_names : list[tuple[str, str | None]]
            List of ``(model_name, version)`` tuples.  ``version`` can be
            ``None`` to load the latest.
        """
        for name, ver in model_names:
            try:
                self.load_model(name, version=ver)
                logger.info("Pre-warmed cache for model %s", name)
            except Exception as exc:
                logger.warning("Failed to pre-warm model %s: %s", name, exc)

    def clear_cache(self) -> None:
        """Clear the entire model cache."""
        with self._mutex:
            self._cache.clear()
            self._cache_access_times.clear()

    # ------------------------------------------------------------------
    # Model Information
    # ------------------------------------------------------------------

    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a registered model.

        Parameters
        ----------
        model_name : str
            Model identifier.
        version : str or None
            Specific version or ``None`` for latest.

        Returns
        -------
        dict
            Model information dictionary.
        """
        with self._mutex:
            versions = self._models.get(model_name)
            if not versions:
                raise KeyError(f"Model '{model_name}' not found.")

            if version is None:
                # Return latest version info
                version_numbers = []
                for v in versions:
                    if v.startswith("v"):
                        try:
                            version_numbers.append((int(v[1:]), v))
                        except ValueError:
                            pass
                if version_numbers:
                    version_numbers.sort(reverse=True)
                    version = version_numbers[0][1]
                else:
                    version = list(versions.keys())[0]

            info = versions.get(version)
            if info is None:
                raise KeyError(
                    f"Version '{version}' not found for model '{model_name}'."
                )

            return info.to_dict()

    def list_models(
        self,
        status: Optional[str] = None,
        tag_filter: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """List all registered models.

        Parameters
        ----------
        status : str or None
            Filter by status.
        tag_filter : dict or None
            Filter by tags (all must match).

        Returns
        -------
        list[dict]
            List of model info dictionaries.
        """
        with self._mutex:
            results: List[Dict[str, Any]] = []
            for model_name, versions in self._models.items():
                for version, info in versions.items():
                    if status and info.status != status:
                        continue
                    if tag_filter:
                        match = all(
                            info.tags.get(k) == v for k, v in tag_filter.items()
                        )
                        if not match:
                            continue
                    results.append(info.to_dict())
            return results

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model.

        Parameters
        ----------
        model_name : str
            Model identifier.

        Returns
        -------
        list[dict]
            Sorted by version (newest first).
        """
        with self._mutex:
            versions = self._models.get(model_name, {})
            version_list = [info.to_dict() for info in versions.values()]
            # Sort by version number descending
            version_list.sort(
                key=lambda x: int(x["version"].lstrip("v"))
                if x["version"].lstrip("v").isdigit()
                else 0,
                reverse=True,
            )
            return version_list

    # ------------------------------------------------------------------
    # Model Lifecycle
    # ------------------------------------------------------------------

    def update_model_status(
        self,
        model_name: str,
        version: str,
        new_status: str,
    ) -> None:
        """Update the status of a registered model.

        Parameters
        ----------
        model_name : str
            Model identifier.
        version : str
            Version string.
        new_status : str
            One of ``"registered"``, ``"staging"``, ``"production"``, ``"archived"``.
        """
        valid = {"registered", "staging", "production", "archived"}
        if new_status not in valid:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {valid}.")

        with self._mutex:
            versions = self._models.get(model_name)
            if not versions or version not in versions:
                raise KeyError(f"Model '{model_name}' version '{version}' not found.")

            versions[version].status = new_status
            versions[version].updated_at = datetime.now(timezone.utc).isoformat()
            self._save_registry()

        # Sync with MLflow
        if self._mlflow_available and new_status in ("staging", "production", "archived"):
            try:
                mlflow_status_map = {
                    "staging": "Staging",
                    "production": "Production",
                    "archived": "Archived",
                }
                self._mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=int(version.lstrip("v")),
                    stage=mlflow_status_map[new_status],
                )
            except Exception as exc:
                logger.warning("MLflow status update failed: %s", exc)

        logger.info("Model %s/%s status updated to %s", model_name, version, new_status)

    def delete_model(self, model_name: str, version: Optional[str] = None) -> None:
        """Delete a model or specific version from the registry.

        Parameters
        ----------
        model_name : str
            Model identifier.
        version : str or None
            Specific version.  ``None`` deletes all versions.
        """
        with self._mutex:
            if model_name not in self._models:
                raise KeyError(f"Model '{model_name}' not found.")

            if version:
                if version not in self._models[model_name]:
                    raise KeyError(
                        f"Version '{version}' not found for model '{model_name}'."
                    )
                del self._models[model_name][version]
                self._cache.pop(f"{model_name}/{version}", None)
                self._cache_access_times.pop(f"{model_name}/{version}", None)

                # Remove from disk
                model_path = os.path.join(
                    self.storage_path, "models", model_name, version
                )
                if os.path.exists(model_path):
                    import shutil
                    shutil.rmtree(model_path, ignore_errors=True)

                if not self._models[model_name]:
                    del self._models[model_name]
            else:
                # Delete all versions
                for ver in list(self._models[model_name].keys()):
                    self._cache.pop(f"{model_name}/{ver}", None)
                    self._cache_access_times.pop(f"{model_name}/{ver}", None)
                    model_path = os.path.join(
                        self.storage_path, "models", model_name, ver
                    )
                    if os.path.exists(model_path):
                        import shutil
                        shutil.rmtree(model_path, ignore_errors=True)
                del self._models[model_name]

            self._save_registry()

        logger.info(
            "Deleted model %s (version=%s)", model_name, version or "all"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_registry(self) -> None:
        """Save registry metadata to disk."""
        meta_path = os.path.join(self.storage_path, "registry.json")
        data = {}
        for name, versions in self._models.items():
            data[name] = {
                ver: info.to_dict() for ver, info in versions.items()
            }

        try:
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to save registry: %s", exc)

    def _load_registry(self) -> None:
        """Load registry metadata from disk."""
        meta_path = os.path.join(self.storage_path, "registry.json")
        if not os.path.exists(meta_path):
            return

        try:
            with open(meta_path, "r") as f:
                data = json.load(f)

            for name, versions in data.items():
                self._models[name] = {}
                for ver, ver_data in versions.items():
                    self._models[name][ver] = ModelInfo.from_dict(ver_data)

            logger.info(
                "Loaded registry with %d models from %s",
                len(self._models),
                meta_path,
            )
        except Exception as exc:
            logger.warning("Failed to load registry from %s: %s", meta_path, exc)
