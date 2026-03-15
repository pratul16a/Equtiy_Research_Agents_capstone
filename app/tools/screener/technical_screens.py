"""Technical / price screens — 200 DMA, volume breakout."""

from __future__ import annotations

import logging
from typing import Any

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
) -> list[dict[str, Any]]:
    """Screen for stocks where 5-day avg volume > 2x 50-day avg volume."""
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
