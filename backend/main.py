"""Main entry point for AI Financial Brain backend server."""

import sys
from pathlib import Path

# Ensure absolute imports like `backend.*` work even when running:
# `cd backend && python main.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from backend.config.settings import get_settings


def main() -> None:
    """Start the AI Financial Brain API server."""
    settings = get_settings()

    # Setup structured logging
    from backend.utils.logging import setup_logging
    setup_logging(
        log_level=settings.log_level,
        json_logs=settings.environment == "production",
    )

    uvicorn.run(
        "backend.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1 if settings.debug else settings.api_workers,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=not settings.debug,
    )


if __name__ == "__main__":
    main()
