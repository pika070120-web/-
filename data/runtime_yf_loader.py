from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd


class RuntimeYFinanceLoader:
    _YF_MAP = {
        "VIX": "^VIX",
    }

    def _normalize_ticker(self, ticker: str) -> str:
        return self._YF_MAP.get(ticker.upper(), ticker.upper())

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
        end_plus = end_date + timedelta(days=1)

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

        if isinstance(raw.columns, pd.MultiIndex):
            flattened = []
            for col in raw.columns:
                if isinstance(col, tuple):
                    flattened.append(col[0])
                else:
                    flattened.append(col)
            raw.columns = flattened

        date_col = None
        for candidate in ["Date", "Datetime"]:
            if candidate in raw.columns:
                date_col = candidate
                break

        if date_col is None:
            return None

        def _col1d(name: str) -> pd.Series:
            value = raw[name]
            if isinstance(value, pd.DataFrame):
                value = value.iloc[:, 0]
            return pd.to_numeric(value, errors="coerce")

        if "Volume" in raw.columns:
            volume_value = raw["Volume"]
            if isinstance(volume_value, pd.DataFrame):
                volume_value = volume_value.iloc[:, 0]
            volume_series = pd.to_numeric(volume_value, errors="coerce").fillna(0.0)
        else:
            volume_series = pd.Series(0.0, index=raw.index)

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col]).dt.date,
                "open": _col1d("Open"),
                "high": _col1d("High"),
                "low": _col1d("Low"),
                "close": _col1d("Close"),
                "volume": volume_series,
            }
        ).dropna(subset=["open", "high", "low", "close"])

        if len(df) == 0:
            return None

        return df.sort_values("date").reset_index(drop=True)

    def load_multiple(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self.load_single(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                out[ticker] = df
        return out

    def get_ndx100_components(self) -> list[str]:
        return [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "META",
            "GOOGL",
            "GOOG",
            "TSLA",
            "COST",
            "NFLX",
            "AMD",
            "ADBE",
            "PEP",
            "CSCO",
            "INTC",
            "QCOM",
            "AMGN",
            "TXN",
            "INTU",
            "HON",
            "AMAT",
            "ISRG",
            "SBUX",
            "ADI",
            "MDLZ",
            "ADP",
            "GILD",
        ]