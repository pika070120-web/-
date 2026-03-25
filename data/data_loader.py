"""
data/data_loader.py

Provides:
- DataLoader base class
- YFinanceDataLoader for real market data
- generate_synthetic_ohlcv() for tests

Used by:
- tests/helpers.py
- runners/main_runner.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _make_business_dates(n_days: int, end_date: Optional[date] = None) -> List[date]:
    """
    Build a list of business dates ending at end_date (or today if omitted).
    """
    end = end_date or date.today()
    dates: List[date] = []
    current = end
    while len(dates) < n_days:
        if current.weekday() < 5:
            dates.append(current)
        current = current.fromordinal(current.toordinal() - 1)
    dates.reverse()
    return dates


def generate_synthetic_ohlcv(
    ticker: str,
    n_days: int = 90,
    start_price: float = 100.0,
    trend: float = 0.001,
    volatility: float = 0.02,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for tests.

    Args:
        ticker: Symbol name (not used in the dataframe, kept for API consistency)
        n_days: Number of trading days
        start_price: Starting close price
        trend: Mean daily return
        volatility: Daily return stddev
        seed: Random seed for deterministic tests
    """
    rng = np.random.default_rng(seed)
    dates = _make_business_dates(n_days)

    closes = np.zeros(n_days, dtype=float)
    closes[0] = max(start_price, 1.0)

    for i in range(1, n_days):
        daily_ret = rng.normal(trend, volatility)
        closes[i] = max(1.0, closes[i - 1] * (1 + daily_ret))

    opens = np.zeros(n_days, dtype=float)
    highs = np.zeros(n_days, dtype=float)
    lows = np.zeros(n_days, dtype=float)
    volumes = rng.integers(500_000, 2_000_000, size=n_days).astype(float)

    opens[0] = closes[0] * (1 + rng.uniform(-0.002, 0.002))
    for i in range(1, n_days):
        opens[i] = closes[i - 1] * (1 + rng.uniform(-0.005, 0.005))

    for i in range(n_days):
        base_high = max(opens[i], closes[i])
        base_low = min(opens[i], closes[i])

        highs[i] = base_high * (1 + abs(rng.normal(0.003, 0.001)))
        lows[i] = max(0.5, base_low * (1 - abs(rng.normal(0.003, 0.001))))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": volumes,
        }
    )
    return df


class DataLoader(ABC):
    """
    Abstract market data loader.
    """

    @abstractmethod
    def load_single(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def load_multiple(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.load_single(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                data[ticker] = df
        return data

    def get_ndx100_components(self) -> List[str]:
        """
        Static fallback universe list.
        This is intentionally a stable hardcoded list for local development.
        """
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
            "TSLA", "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO", "TMUS",
            "INTC", "QCOM", "AMGN", "TXN", "INTU", "HON", "AMAT", "BKNG",
            "ISRG", "SBUX", "ADI", "MDLZ", "ADP", "GILD",
        ]


class YFinanceDataLoader(DataLoader):
    """
    Real market data loader using yfinance.

    Notes:
    - VIX is mapped to ^VIX
    - Returned dataframe columns are standardized to:
      date, open, high, low, close, volume
    """

    _YF_MAP = {
        "VIX": "^VIX",
    }

    def _normalize_ticker(self, ticker: str) -> str:
        return self._YF_MAP.get(ticker, ticker)

    def load_single(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance가 설치되지 않았습니다. 터미널에 pip install yfinance 입력해서 설치하세요."
            ) from exc

        yf_ticker = self._normalize_ticker(ticker)

        # yfinance end date is exclusive in many cases, so add 1 day buffer
        end_plus = end_date.fromordinal(end_date.toordinal() + 1)

        try:
            raw = yf.download(
                yf_ticker,
                start=start_date.isoformat(),
                end=end_plus.isoformat(),
                auto_adjust=False,
                progress=False,
            )
        except Exception:
            return None

        if raw is None or len(raw) == 0:
            return None

        raw = raw.reset_index()

        date_col = None
        for candidate in ["Date", "Datetime"]:
            if candidate in raw.columns:
                date_col = candidate
                break

        if date_col is None:
            return None

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col]).dt.date,
                "open": pd.to_numeric(raw["Open"], errors="coerce"),
                "high": pd.to_numeric(raw["High"], errors="coerce"),
                "low": pd.to_numeric(raw["Low"], errors="coerce"),
                "close": pd.to_numeric(raw["Close"], errors="coerce"),
                "volume": pd.to_numeric(raw.get("Volume", 0), errors="coerce").fillna(0.0),
            }
        ).dropna(subset=["open", "high", "low", "close"])

        if len(df) == 0:
            return None

        return df.sort_values("date").reset_index(drop=True)