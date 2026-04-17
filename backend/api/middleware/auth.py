"""
Authentication Middleware.

Provides JWT verification and API-key authentication for protected endpoints.
Supports rate limiting per user via Redis-backed sliding window.
"""

import hashlib
import hmac
import json
import logging

import time
from typing import Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# In production these come from env / Vault — hardcoded here for clarity
settings = get_settings()
JWT_SECRET = settings.jwt_secret_key
JWT_ALGORITHM = settings.jwt_algorithm
API_KEY_HEADER = "X-API-Key"

# Rate limiter defaults
RATE_LIMIT_REQUESTS = 100  # max requests per window
RATE_LIMIT_WINDOW_S = 60  # window duration in seconds

# Simulated API key store (replace with DB-backed store)
_API_KEYS: Dict[str, Dict[str, str]] = {
    "dev-key-aifin-001": {"user_id": "usr_dev_001", "role": "admin", "tier": "premium"},
    "dev-key-aifin-002": {"user_id": "usr_dev_002", "role": "user", "tier": "standard"},
}

security_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# JWT Helpers (lightweight — no PyJWT dependency required)
# ---------------------------------------------------------------------------

def _base64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(s: str) -> bytes:
    import base64
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def decode_jwt(token: str) -> Dict:
    """
    Decode and verify a JWT token using HMAC-SHA256.

    Returns the payload dict or raises HTTPException on invalid token.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed JWT")

        header_b64, payload_b64, signature_b64 = parts

        header = json.loads(_base64url_decode(header_b64))
        if header.get("alg") != JWT_ALGORITHM:
            raise ValueError(f"Unsupported algorithm: {header.get('alg')}")

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
        actual_sig = _base64url_decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")

        payload = json.loads(_base64url_decode(payload_b64))

        # Check expiration
        exp = payload.get("exp")
        if exp is not None and time.time() > exp:
            raise ValueError("Token expired")

        return payload

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

async def _check_rate_limit(request: Request, user_id: str) -> None:
    """
    Sliding-window rate limiter backed by Redis.

    Raises HTTPException 429 if the user has exceeded their rate limit.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return  # rate limiting disabled if Redis is unavailable

    key = f"ratelimit:{user_id}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_S

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, RATE_LIMIT_WINDOW_S + 10)
    results = await pipe.execute()

    request_count = results[2]

    if request_count > RATE_LIMIT_REQUESTS:
        logger.warning("Rate limit exceeded for user %s (%d requests)", user_id, request_count)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {request_count}/{RATE_LIMIT_REQUESTS} requests in {RATE_LIMIT_WINDOW_S}s",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_S)},
        )


# ---------------------------------------------------------------------------
# Authentication Dependency
# ---------------------------------------------------------------------------

async def get_current_user(request: Request) -> Dict:
    """
    FastAPI dependency that extracts and validates the current user.

    Supports two auth mechanisms:
    1. JWT Bearer token (Authorization: Bearer <token>)
    2. API key (X-API-Key header)

    Returns a dict with at least: {"user_id": str, "auth_method": str}
    """
    # --- Try API Key first ---
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        key_info = _API_KEYS.get(api_key)
        if key_info is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        user_id = key_info["user_id"]
        await _check_rate_limit(request, user_id)
        return {
            "user_id": user_id,
            "role": key_info.get("role", "user"),
            "tier": key_info.get("tier", "standard"),
            "auth_method": "api_key",
        }

    # --- Try JWT Bearer ---
    credentials: Optional[HTTPAuthorizationCredentials] = await security_scheme(request)
    if credentials and credentials.scheme.lower() == "bearer":
        payload = decode_jwt(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim",
            )
        await _check_rate_limit(request, user_id)
        return {
            "user_id": user_id,
            "role": payload.get("role", "user"),
            "tier": payload.get("tier", "standard"),
            "email": payload.get("email"),
            "auth_method": "jwt",
        }

    # --- No auth provided ---
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication credentials. Provide Authorization: Bearer <token> or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(*roles: str) -> Callable:
    """Dependency factory that enforces role-based access control."""

    async def role_checker(request: Request) -> Dict:
        user = await get_current_user(request)
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {roles}",
            )
        return user

    return role_checker


