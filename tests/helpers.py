"""
tests/helpers.py
Standalone test helper functions.
NOT a pytest conftest — importable directly in test files.
Conftest.py imports from here; test files import from here directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional


def make_bull_df(n: int = 90, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Steady uptrend with moderate volatility."""
    from data.data_loader import generate_synthetic_ohlcv
    return generate_synthetic_ohlcv(
        "TEST",
        n_days=n,
        start_price=start_price,
        trend=0.003,
        volatility=0.012,
        seed=seed,
    )


def make_bear_df(n: int = 90, start_price: float = 200.0, seed: int = 42) -> pd.DataFrame:
    """Steady downtrend."""
    from data.data_loader import generate_synthetic_ohlcv
    return generate_synthetic_ohlcv(
        "TEST",
        n_days=n,
        start_price=start_price,
        trend=-0.003,
        volatility=0.018,
        seed=seed,
    )


def make_flat_df(n: int = 90, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Flat/no-pullback data: gate3 will FAIL (drawdown too shallow)."""
    from data.data_loader import generate_synthetic_ohlcv
    return generate_synthetic_ohlcv(
        "TEST",
        n_days=n,
        start_price=start_price,
        trend=0.005,
        volatility=0.002,
        seed=seed,
    )


def make_pullback_df(seed: int = 42) -> pd.DataFrame:
    """
    Valid pullback (5-8%), standard quality.
    Gate3: PASS range (human review needed, prelim=PASS).
    Gate4: slight recovery (may pass with low momentum).
    """
    n = 90
    np.random.seed(seed)
    base_dates = _make_dates(n)

    closes = np.zeros(n)
    closes[0] = 100.0
    for i in range(1, n - 20):
        closes[i] = closes[i - 1] * (1 + np.random.normal(0.002, 0.010))

    peak_idx = n - 21
    peak = float(closes[peak_idx])

    for i in range(n - 20, n - 5):
        progress = (i - (n - 20)) / 15
        closes[i] = peak * (1 - 0.07 * progress)

    for i in range(n - 5, n):
        closes[i] = closes[n - 6] * (1 + 0.004 * (i - (n - 6)))

    closes = np.maximum(closes, 1.0)
    return _build_df(base_dates, closes, vol_scale=1.0)


def make_no_recovery_df(seed: int = 55) -> pd.DataFrame:
    """
    Valid pullback range but NO recovery — gate4 will FAIL (negative momentum).
    Used to test that human cannot resurrect gate4 FAIL.
    """
    n = 90
    np.random.seed(seed)
    base_dates = _make_dates(n)

    closes = np.zeros(n)
    closes[0] = 100.0
    for i in range(1, n - 25):
        closes[i] = closes[i - 1] * (1 + np.random.normal(0.002, 0.008))

    peak = float(closes[n - 26])

    for i in range(n - 25, n):
        progress = (i - (n - 26)) / 25
        closes[i] = peak * (1 - 0.10 * progress)

    closes = np.maximum(closes, 1.0)
    return _build_df(base_dates, closes, vol_scale=1.0)


def make_premium_pullback_df(seed: int = 99) -> pd.DataFrame:
    """
    Premium-quality pullback:
      - Tight drawdown (<= 8%)
      - Volume contracting during pullback (score +1)
      - Near 20-SMA at trough (score +1)
    Expected gate3 result: PREMIUM_PASS (score >= 2).
    """
    n = 90
    np.random.seed(seed)
    base_dates = _make_dates(n)

    closes = np.zeros(n)
    closes[0] = 200.0
    for i in range(1, n - 20):
        closes[i] = closes[i - 1] * (1 + np.random.normal(0.003, 0.008))

    peak = float(closes[n - 21])

    for i in range(n - 20, n - 5):
        progress = (i - (n - 20)) / 15
        closes[i] = peak * (1 - 0.06 * progress)

    for i in range(n - 5, n):
        closes[i] = closes[n - 6] * (1 + 0.003 * (i - (n - 6)))

    closes = np.maximum(closes, 1.0)

    volumes = np.full(n, 1_200_000.0)
    for i in range(n - 20, n):
        volumes[i] = 700_000.0 + np.random.randint(0, 50_000)

    return _build_df(base_dates, closes, vol_scale=1.0, custom_volume=volumes)


def make_premium_restrength_df(seed: int = 77) -> pd.DataFrame:
    """
    Premium-quality re-strengthening:
      - Momentum >= 3% over last 5 bars (score +1)
      - Volume expanding on recovery >= 1.2x (score +1)
      - Price above 10-SMA (score +1)
    Expected gate4 result: PREMIUM_PASS (score >= 2).
    """
    n = 90
    np.random.seed(seed)
    base_dates = _make_dates(n)

    closes = np.zeros(n)
    closes[0] = 150.0
    for i in range(1, n - 15):
        closes[i] = closes[i - 1] * (1 + np.random.normal(0.002, 0.010))

    bottom = float(closes[n - 16])

    for i in range(n - 15, n - 5):
        closes[i] = bottom * (1 - 0.06 * (i - (n - 15)) / 10)

    recovery_start = float(closes[n - 6])
    for i in range(n - 5, n):
        closes[i] = recovery_start * (1 + 0.008 * (i - (n - 6)))

    closes = np.maximum(closes, 1.0)

    volumes = np.full(n, 1_000_000.0)
    for i in range(n - 15, n - 5):
        volumes[i] = 600_000.0
    for i in range(n - 5, n):
        volumes[i] = 1_600_000.0

    return _build_df(base_dates, closes, vol_scale=1.0, custom_volume=volumes)


def _make_dates(n: int, start: Optional[date] = None):
    d = start or date(2023, 1, 3)
    dates = []
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _build_df(
    dates,
    closes: np.ndarray,
    vol_scale: float = 1.0,
    custom_volume: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    n = len(closes)
    volumes = custom_volume if custom_volume is not None else (
        np.random.randint(500_000, 2_000_000, n).astype(float) * vol_scale
    )
    return pd.DataFrame({
        "date": dates[:n],
        "open": np.round(closes * (1 + np.random.uniform(-0.002, 0.002, n)), 2),
        "high": np.round(closes * (1 + np.abs(np.random.normal(0.003, 0.001, n))), 2),
        "low": np.round(closes * (1 - np.abs(np.random.normal(0.003, 0.001, n))), 2),
        "close": np.round(closes, 2),
        "volume": volumes,
    })