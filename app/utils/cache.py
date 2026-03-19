"""TTL-based in-memory cache, SQLite disk cache, and retry utilities for API calls."""

from __future__ import annotations

import functools
import json
import logging
import os
import pickle
import sqlite3
import time
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── TTL In-Memory Cache ──────────────────────────────────────

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = Lock()

DEFAULT_TTL = 300  # 5 minutes


def cache_get(key: str) -> Any | None:
    """Get a value from in-memory cache, then fall back to disk cache."""
    # Check in-memory first
    with _cache_lock:
        if key in _cache:
            expiry, value = _cache[key]
            if time.time() < expiry:
                return value
            del _cache[key]

    # Fall back to disk cache
    disk_val = disk_cache_get(key)
    if disk_val is not None:
        # Promote to in-memory cache
        with _cache_lock:
            _cache[key] = (time.time() + DEFAULT_TTL, disk_val)
        return disk_val

    return None


def cache_set(key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
    """Store a value in both in-memory and disk cache."""
    with _cache_lock:
        _cache[key] = (time.time() + ttl, value)
    # Also persist to disk with longer TTL
    disk_cache_set(key, value, ttl=max(ttl, DISK_CACHE_TTL))


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


# ── SQLite Disk Cache ────────────────────────────────────────

DISK_CACHE_TTL = 14400  # 4 hours — survives process restarts
_DISK_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "cache.db",
)
_disk_cache_lock = Lock()
_disk_conn: sqlite3.Connection | None = None


def _get_disk_conn() -> sqlite3.Connection:
    """Get or create the SQLite connection (lazy init)."""
    global _disk_conn
    if _disk_conn is None:
        os.makedirs(os.path.dirname(_DISK_CACHE_PATH), exist_ok=True)
        _disk_conn = sqlite3.connect(_DISK_CACHE_PATH, check_same_thread=False)
        _disk_conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value BLOB, expiry REAL)"
        )
        _disk_conn.execute("CREATE INDEX IF NOT EXISTS idx_expiry ON cache(expiry)")
        _disk_conn.commit()
    return _disk_conn


def disk_cache_get(key: str) -> Any | None:
    """Get a value from the SQLite disk cache."""
    try:
        with _disk_cache_lock:
            conn = _get_disk_conn()
            row = conn.execute(
                "SELECT value, expiry FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value_blob, expiry = row
            if time.time() > expiry:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return None
            return pickle.loads(value_blob)
    except Exception as e:
        logger.debug("Disk cache get failed for %s: %s", key, e)
        return None


def disk_cache_set(key: str, value: Any, ttl: int = DISK_CACHE_TTL) -> None:
    """Store a value in the SQLite disk cache."""
    try:
        # Pickle OUTSIDE the lock to avoid blocking other threads
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with _disk_cache_lock:
            conn = _get_disk_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expiry) VALUES (?, ?, ?)",
                (key, blob, time.time() + ttl),
            )
            conn.commit()
    except Exception as e:
        logger.debug("Disk cache set failed for %s: %s", key, e)


def disk_cache_cleanup() -> int:
    """Remove expired entries from disk cache. Returns count removed."""
    try:
        with _disk_cache_lock:
            conn = _get_disk_conn()
            cursor = conn.execute("DELETE FROM cache WHERE expiry < ?", (time.time(),))
            conn.commit()
            return cursor.rowcount
    except Exception:
        return 0


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

    def reset(self) -> None:
        """Reset state for a fresh run (avoids stale timestamps)."""
        with self._lock:
            self._last_call = 0.0


# Global rate limiter for yfinance calls (10 calls/sec — raised from 5)
yfinance_rate_limiter = RateLimiter(calls_per_second=10.0)


# ── Circuit Breaker ──────────────────────────────────────────

class CircuitBreaker:
    """Trip after `threshold` consecutive failures; auto-reset after `reset_seconds`."""

    def __init__(self, threshold: int = 3, reset_seconds: float = 300.0):
        self._threshold = threshold
        self._reset_seconds = reset_seconds
        self._failures = 0
        self._tripped_at: float | None = None
        self._lock = Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._tripped_at is None:
                return False
            if time.time() - self._tripped_at > self._reset_seconds:
                self._failures = 0
                self._tripped_at = None
                return False
            return True

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._tripped_at = time.time()

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._tripped_at = None


# Global circuit breaker for BSE API (trips after 3 failures, resets in 5min)
bse_circuit_breaker = CircuitBreaker(threshold=3, reset_seconds=300.0)


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
