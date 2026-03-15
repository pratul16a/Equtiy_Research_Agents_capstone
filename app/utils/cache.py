"""TTL-based in-memory cache and retry utilities for API calls."""

from __future__ import annotations

import functools
import logging
import time
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── TTL Cache ────────────────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = Lock()

DEFAULT_TTL = 300  # 5 minutes


def cache_get(key: str) -> Any | None:
    """Get a value from the cache if it exists and hasn't expired."""
    with _cache_lock:
        if key in _cache:
            expiry, value = _cache[key]
            if time.time() < expiry:
                return value
            del _cache[key]
    return None


def cache_set(key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
    """Store a value in the cache with a TTL in seconds."""
    with _cache_lock:
        _cache[key] = (time.time() + ttl, value)


def cache_clear() -> None:
    """Clear all cached entries."""
    with _cache_lock:
        _cache.clear()


def cache_stats() -> dict:
    """Return cache statistics."""
    with _cache_lock:
        now = time.time()
        total = len(_cache)
        active = sum(1 for exp, _ in _cache.values() if now < exp)
        return {"total_entries": total, "active_entries": active, "expired_entries": total - active}


def cached(ttl: int = DEFAULT_TTL, key_prefix: str = ""):
    """Decorator that caches function results based on arguments.

    Args:
        ttl: Time-to-live in seconds (default 300s / 5 min).
        key_prefix: Optional prefix for the cache key.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build a cache key from function name + args
            parts = [key_prefix or func.__name__]
            parts.extend(str(a) for a in args)
            parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(parts)

            result = cache_get(key)
            if result is not None:
                logger.debug("Cache HIT: %s", key)
                return result

            logger.debug("Cache MISS: %s", key)
            result = func(*args, **kwargs)
            cache_set(key, result, ttl)
            return result

        wrapper.cache_key_prefix = key_prefix or func.__name__
        return wrapper
    return decorator


# ── Retry with Exponential Backoff ───────────────────────────

def retry_on_error(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
):
    """Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds between retries.
        max_delay: Maximum delay cap in seconds.
        exceptions: Tuple of exception types to catch and retry on.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            "Retry %d/%d for %s after error: %s (waiting %.1fs)",
                            attempt + 1, max_retries, func.__name__, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            max_retries, func.__name__, e,
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper
    return decorator


# ── Rate Limiter ─────────────────────────────────────────────

class RateLimiter:
    """Simple token-bucket rate limiter.

    Args:
        calls_per_second: Maximum calls allowed per second.
    """

    def __init__(self, calls_per_second: float = 2.0):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        """Block until it's safe to make the next call."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()


# Global rate limiter for yfinance calls (2 calls/sec max)
yfinance_rate_limiter = RateLimiter(calls_per_second=2.0)


def get_yf_ticker(ticker: str):
    """Get a yfinance Ticker object with rate limiting and caching of the info dict.

    Returns:
        (yf.Ticker, info_dict) tuple. The info dict is cached for 5 minutes.
    """
    import yfinance as yf

    yfinance_rate_limiter.wait()
    stock = yf.Ticker(ticker)

    # Cache the .info call since it's the most expensive
    info_key = f"yf_info:{ticker}"
    info = cache_get(info_key)
    if info is None:
        info = stock.info
        cache_set(info_key, info, ttl=DEFAULT_TTL)

    return stock, info
