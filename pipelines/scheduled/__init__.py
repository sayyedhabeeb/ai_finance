"""
Pipeline scheduler using Prefect.

Defines Prefect flows for each data pipeline, configures cron-like
schedules, and handles error notifications.
"""

from pipelines.scheduled.scheduler import PipelineScheduler

__all__ = ["PipelineScheduler"]
