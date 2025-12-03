"""
Middleware for rate limiting and authentication.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.utils.logging_config import get_logger

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting storage (in-memory, can be upgraded to Redis for distributed systems)
_rate_limit_storage: dict[str, list[float]] = defaultdict(list)
# Track last cleanup time for memory management
_last_cleanup_time: float = 0.0
# Cleanup interval in seconds (clean up stale entries every 5 minutes)
_CLEANUP_INTERVAL: float = 300.0

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""

    def __init__(self, app: Any, requests_per_minute: int = 60) -> None:
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application.
            requests_per_minute: Maximum requests per minute per client (default: 60).
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Process request with rate limiting."""
        global _last_cleanup_time

        # Skip rate limiting for health and metrics endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client identifier (IP address or API key)
        client_id = self._get_client_id(request)

        # Check rate limit
        current_time = time.time()

        # Periodic cleanup of stale entries to prevent memory leak
        if current_time - _last_cleanup_time > _CLEANUP_INTERVAL:
            self._cleanup_stale_entries(current_time)
            _last_cleanup_time = current_time

        client_requests = _rate_limit_storage[client_id]

        # Remove requests outside the time window
        client_requests[:] = [
            req_time
            for req_time in client_requests
            if current_time - req_time < self.window_seconds
        ]

        # Check if limit exceeded
        if len(client_requests) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": int(self.window_seconds - (current_time - client_requests[0])),
                },
            )

        # Add current request
        client_requests.append(current_time)

        # Process request
        response = await call_next(request)
        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting.

        Args:
            request: FastAPI request object.

        Returns:
            Client identifier (API key if present, otherwise IP address).
        """
        # Try to get API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"

        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _cleanup_stale_entries(self, current_time: float) -> None:
        """Remove stale entries from rate limit storage to prevent memory leak.

        Args:
            current_time: Current timestamp.
        """
        stale_clients = []
        for client_id, requests in _rate_limit_storage.items():
            # Remove old requests
            valid_requests = [
                req_time for req_time in requests if current_time - req_time < self.window_seconds
            ]
            if not valid_requests:
                stale_clients.append(client_id)
            else:
                _rate_limit_storage[client_id] = valid_requests

        # Remove clients with no recent requests
        for client_id in stale_clients:
            del _rate_limit_storage[client_id]

        if stale_clients:
            logger.debug(f"Cleaned up {len(stale_clients)} stale rate limit entries")


async def verify_api_key(request: Request) -> str | None:
    """Verify API key from request header.

    Args:
        request: FastAPI request object.

    Returns:
        API key if valid, None otherwise.

    Raises:
        HTTPException: If API key is required but missing or invalid.
    """
    # Check if API key authentication is enabled
    api_key_required = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    if not api_key_required:
        return None

    # Get API key from header
    api_key = await api_key_header(request)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Provide it in the X-API-Key header.",
        )

    # Validate API key
    valid_api_keys = os.getenv("API_KEYS", "").split(",")
    valid_api_keys = [key.strip() for key in valid_api_keys if key.strip()]

    if not valid_api_keys:
        # If API_KEY_REQUIRED is True but no keys are configured, allow all (development mode)
        logger.warning(
            "API_KEY_REQUIRED is True but no API_KEYS configured. "
            "Allowing all requests (development mode)."
        )
        return api_key

    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    return api_key
