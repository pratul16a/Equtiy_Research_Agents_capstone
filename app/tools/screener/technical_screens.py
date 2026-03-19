"""Technical / price screens — 200 DMA, volume breakout, RSI, weekly MA alignment."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from app.utils.cache import cache_get, cache_set, yfinance_rate_limiter

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


def screen_above_200dma(
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with current price above 200-day moving average."""
    results: list[dict] = []

    for ticker in price_df.columns:
        try:
            series = price_df[ticker].dropna()
            if len(series) < 200:
                continue

            dma_200 = series.rolling(window=200).mean()
            current_price = float(series.iloc[-1])
            current_dma = float(dma_200.iloc[-1])

            if pd.isna(current_dma) or current_dma <= 0:
                continue

            if current_price > current_dma:
                pct_above = round((current_price / current_dma - 1) * 100, 2)
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "current_price": round(current_price, 2),
                    "dma_200": round(current_dma, 2),
                    "pct_above_200dma": pct_above,
                    "passed": True,
                })
        except Exception as e:
            logger.debug("200 DMA check failed for %s: %s", ticker, e)

    logger.info("200 DMA screen: %d stocks above 200 DMA", len(results))
    return results


def _download_volume_batch(tickers: list[str], period: str = "3mo") -> pd.DataFrame:
    """Download volume data for a batch of tickers."""
    yfinance_rate_limiter.wait()
    raw = yf.download(tickers, period=period, group_by="ticker", threads=True, progress=False)

    volume_data: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                series = raw["Volume"]
            else:
                series = raw[(ticker, "Volume")]
            if series is not None and not series.dropna().empty:
                volume_data[ticker] = series
        except (KeyError, TypeError):
            pass

    return pd.DataFrame(volume_data)


def screen_volume_breakout(
    tickers: list[str],
    sector_map: dict[str, str],
    preloaded_price_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Screen for stocks where 5-day avg volume > 2x 50-day avg volume."""
    # Try to extract volume from pre-downloaded price_df to avoid duplicate downloads
    volume_df = None
    if preloaded_price_df is not None:
        try:
            if isinstance(preloaded_price_df.columns, pd.MultiIndex):
                vol = preloaded_price_df.xs("Volume", level=0, axis=1)
            elif "Volume" in preloaded_price_df.columns:
                vol = preloaded_price_df[["Volume"]]
            else:
                vol = None
            if vol is not None and not vol.empty:
                volume_df = vol
        except Exception:
            pass

    if volume_df is None:
        cache_key = f"volume_data:{len(tickers)}"
        volume_df = cache_get(cache_key)

        if volume_df is None:
            batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
            frames: list[pd.DataFrame] = []
            for batch in batches:
                df = _download_volume_batch(batch)
                if not df.empty:
                    frames.append(df)
            volume_df = pd.concat(frames, axis=1) if frames else pd.DataFrame()
            if not volume_df.empty:
                volume_df = volume_df.loc[:, ~volume_df.columns.duplicated()]
                cache_set(cache_key, volume_df, ttl=600)

    results: list[dict] = []
    for ticker in volume_df.columns:
        try:
            series = volume_df[ticker].dropna()
            if len(series) < 50:
                continue

            avg_5d = float(series.tail(5).mean())
            avg_50d = float(series.tail(50).mean())

            if avg_50d <= 0:
                continue

            volume_ratio = avg_5d / avg_50d
            if volume_ratio > 2.0:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "avg_volume_5d": int(avg_5d),
                    "avg_volume_50d": int(avg_50d),
                    "volume_ratio": round(volume_ratio, 2),
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Volume breakout check failed for %s: %s", ticker, e)

    logger.info("Volume breakout screen: %d stocks with breakout", len(results))
    return results


# ── RSI Functions ────────────────────────────────────────────


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index using exponential moving average.

    Args:
        series: Price series (Close prices).
        period: RSI lookback period (default 14).

    Returns:
        RSI series (0-100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def screen_rsi_range(
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
    lo: float = 40,
    hi: float = 70,
    period: int = 14,
) -> list[dict[str, Any]]:
    """Screen for stocks with RSI between lo and hi (momentum sweet spot).

    Used by Category B: RSI 40-70 = not overbought, not oversold, trending.
    """
    results: list[dict] = []

    for ticker in price_df.columns:
        try:
            series = price_df[ticker].dropna()
            if len(series) < period + 10:
                continue

            rsi = compute_rsi(series, period)
            current_rsi = float(rsi.iloc[-1])

            if pd.isna(current_rsi):
                continue

            if lo <= current_rsi <= hi:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "rsi": round(current_rsi, 2),
                    "rsi_range": f"{lo}-{hi}",
                    "passed": True,
                })
        except Exception as e:
            logger.debug("RSI range check failed for %s: %s", ticker, e)

    logger.info("RSI range screen (%s-%s): %d stocks passed", lo, hi, len(results))
    return results


def screen_rsi_oversold(
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
    threshold: float = 40,
    period: int = 14,
) -> list[dict[str, Any]]:
    """Screen for stocks with RSI below threshold (oversold / value territory).

    Used by Category C: RSI < 40 = beaten down.
    """
    results: list[dict] = []

    for ticker in price_df.columns:
        try:
            series = price_df[ticker].dropna()
            if len(series) < period + 10:
                continue

            rsi = compute_rsi(series, period)
            current_rsi = float(rsi.iloc[-1])

            if pd.isna(current_rsi):
                continue

            if current_rsi < threshold:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "rsi": round(current_rsi, 2),
                    "threshold": threshold,
                    "passed": True,
                })
        except Exception as e:
            logger.debug("RSI oversold check failed for %s: %s", ticker, e)

    logger.info("RSI oversold screen (<%s): %d stocks passed", threshold, len(results))
    return results


# ── Weekly MA Alignment ──────────────────────────────────────


def _download_weekly_batch(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download weekly close prices for a batch of tickers."""
    yfinance_rate_limiter.wait()
    raw = yf.download(tickers, period=period, interval="1wk", group_by="ticker", threads=True, progress=False)

    close_data: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                series = raw["Close"]
            else:
                series = raw[(ticker, "Close")]
            if series is not None and not series.dropna().empty:
                close_data[ticker] = series
        except (KeyError, TypeError):
            pass

    return pd.DataFrame(close_data)


def screen_weekly_ma_alignment(
    tickers: list[str],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks where price > 10W MA > 20W MA > 40W MA (bullish alignment).

    Used by Category B: strong uptrend confirmation.
    """
    cache_key = f"weekly_prices:{len(tickers)}"
    weekly_df = cache_get(cache_key)

    if weekly_df is None:
        batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
        frames: list[pd.DataFrame] = []
        for batch in batches:
            df = _download_weekly_batch(batch)
            if not df.empty:
                frames.append(df)
        weekly_df = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        if not weekly_df.empty:
            weekly_df = weekly_df.loc[:, ~weekly_df.columns.duplicated()]
            cache_set(cache_key, weekly_df, ttl=600)

    results: list[dict] = []
    for ticker in weekly_df.columns:
        try:
            series = weekly_df[ticker].dropna()
            if len(series) < 40:
                continue

            current_price = float(series.iloc[-1])
            ma_10w = float(series.rolling(10).mean().iloc[-1])
            ma_20w = float(series.rolling(20).mean().iloc[-1])
            ma_40w = float(series.rolling(40).mean().iloc[-1])

            if any(pd.isna(v) for v in [ma_10w, ma_20w, ma_40w]):
                continue

            # Bullish alignment: Price > 10W > 20W > 40W
            if current_price > ma_10w > ma_20w > ma_40w:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "current_price": round(current_price, 2),
                    "ma_10w": round(ma_10w, 2),
                    "ma_20w": round(ma_20w, 2),
                    "ma_40w": round(ma_40w, 2),
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Weekly MA alignment check failed for %s: %s", ticker, e)

    logger.info("Weekly MA alignment screen: %d stocks passed", len(results))
    return results
