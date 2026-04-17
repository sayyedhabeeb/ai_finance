"""Common utility functions for AI Financial Brain."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, TypeVar

T = TypeVar("T")


def generate_id() -> str:
    """Generate a unique ID (UUID4)."""
    return str(uuid.uuid4())


def generate_short_id() -> str:
    """Generate a short unique ID (first 8 chars of UUID4)."""
    return uuid.uuid4().hex[:8]


def utcnow() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def to_iso(dt: datetime | None) -> str | None:
    """Convert datetime to ISO 8601 string."""
    if dt is None:
        return None
    return dt.isoformat()


def from_iso(s: str | None) -> datetime | None:
    """Parse ISO 8601 string to datetime."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


def compute_checksum(data: str | bytes) -> str:
    """Compute SHA-256 checksum for deduplication."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_length, preserving word boundaries."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length - len(suffix)]
    # Try to break at the last space
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.7:
        truncated = truncated[:last_space]
    return truncated + suffix


def safe_json_parse(text: str, default: Any = None) -> Any:
    """Safely parse JSON string, returning default on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dump(obj: Any, default: str = "{}") -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return default


def chunk_list(lst: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def format_currency(value: float, currency: str = "INR", decimals: int = 2) -> str:
    """Format a number as currency string."""
    symbols = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency.upper(), currency)
    return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage string."""
    return f"{value:.{decimals}f}%"


def format_large_number(value: float) -> str:
    """Format large numbers with suffixes (K, M, B, Cr, L)."""
    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if abs_val >= 1e7:  # 1 Crore
        return f"{sign}₹{abs_val / 1e7:.2f}Cr"
    elif abs_val >= 1e5:  # 1 Lakh
        return f"{sign}₹{abs_val / 1e5:.2f}L"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val / 1e3:.2f}K"
    else:
        return f"{sign}{abs_val:.2f}"
