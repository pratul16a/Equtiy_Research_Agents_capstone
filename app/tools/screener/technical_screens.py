"""Technical / price screens — 200 DMA, golden alignment, MACD, OBV, volume breakout, RSI, weekly MA."""

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

            pct_above = round((current_price / current_dma - 1) * 100, 2)
            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "current_price": round(current_price, 2),
                "dma_200": round(current_dma, 2),
                "pct_above_200dma": pct_above,
                "passed": current_price > current_dma,
            })
        except Exception as e:
            logger.debug("200 DMA check failed for %s: %s", ticker, e)

    logger.info("200 DMA screen: %d stocks above 200 DMA", sum(1 for r in results if r["passed"]))
    return results


# ── Golden Alignment (50 DMA > 200 DMA) ────────────────────


def screen_golden_alignment(
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks where Price > 50 DMA > 200 DMA (golden cross alignment).

    All timeframes aligned bullish — stronger than just being above 200 DMA.
    """
    results: list[dict] = []

    for ticker in price_df.columns:
        try:
            series = price_df[ticker].dropna()
            if len(series) < 200:
                continue

            current_price = float(series.iloc[-1])
            dma_50 = float(series.rolling(50).mean().iloc[-1])
            dma_200 = float(series.rolling(200).mean().iloc[-1])

            if any(pd.isna(v) or v <= 0 for v in [dma_50, dma_200]):
                continue

            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "current_price": round(current_price, 2),
                "dma_50": round(dma_50, 2),
                "dma_200": round(dma_200, 2),
                "passed": current_price > dma_50 > dma_200,
            })
        except Exception as e:
            logger.debug("Golden alignment check failed for %s: %s", ticker, e)

    logger.info("Golden alignment screen: %d stocks passed", sum(1 for r in results if r["passed"]))
    return results


# ── Near 52-Week High ───────────────────────────────────────


def screen_near_52w_high(
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
    threshold: float = 0.10,
) -> list[dict[str, Any]]:
    """Screen for stocks within `threshold` (default 10%) of their 52-week high.

    Near highs = strength, not weakness. Breakout candidates.
    """
    results: list[dict] = []

    for ticker in price_df.columns:
        try:
            series = price_df[ticker].dropna()
            if len(series) < 200:
                continue

            high_52w = float(series.tail(252).max())
            current_price = float(series.iloc[-1])

            if high_52w <= 0:
                continue

            pct_from_high = (high_52w - current_price) / high_52w
            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "current_price": round(current_price, 2),
                "high_52w": round(high_52w, 2),
                "pct_from_high": round(pct_from_high * 100, 2),
                "passed": pct_from_high <= threshold,
            })
        except Exception as e:
            logger.debug("Near 52W high check failed for %s: %s", ticker, e)

    logger.info("Near 52W high screen (within %s%%): %d stocks passed", threshold * 100, sum(1 for r in results if r["passed"]))
    return results


# ── MACD Bullish ────────────────────────────────────────────


def screen_macd_bullish(
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> list[dict[str, Any]]:
    """Screen for stocks where MACD line is above signal line (bullish momentum).

    MACD(12,26,9) — standard parameters.
    """
    results: list[dict] = []

    for ticker in price_df.columns:
        try:
            series = price_df[ticker].dropna()
            if len(series) < slow + signal + 10:
                continue

            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            macd_val = float(macd_line.iloc[-1])
            signal_val = float(signal_line.iloc[-1])
            hist_val = float(histogram.iloc[-1])

            if pd.isna(macd_val) or pd.isna(signal_val):
                continue

            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "macd": round(macd_val, 2),
                "signal_line": round(signal_val, 2),
                "histogram": round(hist_val, 2),
                "passed": macd_val > signal_val,
            })
        except Exception as e:
            logger.debug("MACD bullish check failed for %s: %s", ticker, e)

    logger.info("MACD bullish screen: %d stocks passed", sum(1 for r in results if r["passed"]))
    return results


# ── OBV Rising ──────────────────────────────────────────────


def screen_obv_rising(
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
    volume_df: pd.DataFrame | None = None,
    lookback: int = 20,
) -> list[dict[str, Any]]:
    """Screen for stocks where On-Balance Volume is rising over `lookback` days.

    Cumulative volume confirms price trend isn't hollow.
    Needs volume data — pass `volume_df` or it will be downloaded.
    """
    # Get volume data
    if volume_df is None:
        tickers = list(price_df.columns)
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

    for ticker in price_df.columns:
        try:
            if ticker not in volume_df.columns:
                continue

            price_series = price_df[ticker].dropna()
            vol_series = volume_df[ticker].dropna()

            # Align on common dates
            common = price_series.index.intersection(vol_series.index)
            if len(common) < lookback + 5:
                continue

            prices = price_series.loc[common]
            volumes = vol_series.loc[common]

            # Compute OBV
            price_change = prices.diff()
            signed_volume = volumes.copy()
            signed_volume[price_change < 0] = -signed_volume[price_change < 0]
            signed_volume[price_change == 0] = 0
            obv = signed_volume.cumsum()

            obv_now = float(obv.iloc[-1])
            obv_ago = float(obv.iloc[-lookback])

            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "obv_current": int(obv_now),
                "obv_20d_ago": int(obv_ago),
                "obv_change_pct": round((obv_now - obv_ago) / max(abs(obv_ago), 1) * 100, 2),
                "passed": obv_now > obv_ago,
            })
        except Exception as e:
            logger.debug("OBV rising check failed for %s: %s", ticker, e)

    logger.info("OBV rising screen (%dd): %d stocks passed", lookback, sum(1 for r in results if r["passed"]))
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
    preloaded_volume_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Screen for stocks where 5-day avg volume > 2x 50-day avg volume."""
    volume_df = preloaded_volume_df

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
            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "avg_volume_5d": int(avg_5d),
                "avg_volume_50d": int(avg_50d),
                "volume_ratio": round(volume_ratio, 2),
                "passed": volume_ratio > 2.0,
            })
        except Exception as e:
            logger.debug("Volume breakout check failed for %s: %s", ticker, e)

    logger.info("Volume breakout screen: %d stocks with breakout", sum(1 for r in results if r["passed"]))
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

            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "rsi": round(current_rsi, 2),
                "rsi_range": f"{lo}-{hi}",
                "passed": lo <= current_rsi <= hi,
            })
        except Exception as e:
            logger.debug("RSI range check failed for %s: %s", ticker, e)

    logger.info("RSI range screen (%s-%s): %d stocks passed", lo, hi, sum(1 for r in results if r["passed"]))
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

            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "rsi": round(current_rsi, 2),
                "threshold": threshold,
                "passed": current_rsi < threshold,
            })
        except Exception as e:
            logger.debug("RSI oversold check failed for %s: %s", ticker, e)

    logger.info("RSI oversold screen (<%s): %d stocks passed", threshold, sum(1 for r in results if r["passed"]))
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
