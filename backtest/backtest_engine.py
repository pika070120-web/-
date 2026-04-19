# backtest/backtest_engine.py  (v1.0)
"""
백테스트 엔진
- 기존 MarketFilter / StockEngine / ETFEngine 재사용
- 날짜별 데이터 슬라이싱 → 신호 생성 → 가상 포트폴리오 운용
- 손절가 hit 또는 FULL_EXIT 신호 시 청산
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.models import (
    AccountState,
    ActionMode,
    EntryClass,
    MarketFilterResult,
    StockCandidate,
    WeeklyStockPool,
)
from filters.market_filter import MarketFilter
from filters.etf_engine import ETFEngine
from strategies.stock_engine import StockEngine

logger = logging.getLogger(__name__)


# ── 백테스트 포지션 ────────────────────────────────────────────────────────────

@dataclass
class BTPosition:
    ticker: str
    entry_price: float
    stop_loss: float
    shares: float
    entry_date: date
    entry_class: str
    exposure_pct: float
    highest_price: float = 0.0  # 트레일링 스탑용 최고가 추적


# ── 백테스트 일일 결과 ─────────────────────────────────────────────────────────

@dataclass
class BTDayResult:
    trade_date: date
    market_state: str
    action_mode: str
    new_entries: List[str]
    exits: List[str]
    portfolio_value: float
    cash: float
    total_value: float
    daily_pnl_pct: float


# ── 백테스트 최종 결과 ─────────────────────────────────────────────────────────

@dataclass
class BTSummary:
    start_date: date
    end_date: date
    initial_capital: float
    final_value: float
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    daily_results: List[BTDayResult] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)


# ── 데이터 슬라이서 ────────────────────────────────────────────────────────────

def slice_data(
    full_data: Dict[str, pd.DataFrame],
    as_of: date,
    min_rows: int = 60,
) -> Dict[str, pd.DataFrame]:
    """as_of 날짜까지의 데이터만 잘라서 반환."""
    sliced: Dict[str, pd.DataFrame] = {}
    for ticker, df in full_data.items():
        sub = df[df["date"] <= as_of].copy()
        if len(sub) >= min_rows:
            sliced[ticker] = sub.reset_index(drop=True)
    return sliced


def get_trading_days(
    full_data: Dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
) -> List[date]:
    """QQQ 기준 실제 거래일 목록 추출."""
    ref = full_data.get("QQQ") if full_data.get("QQQ") is not None else full_data.get("SPY")
    if ref is None:
        return []
    days = [
        d for d in ref["date"].tolist()
        if start_date <= d <= end_date
    ]
    return sorted(set(days))


def compute_stop_loss(df: pd.DataFrame, atr_multiple: float = 1.5) -> Optional[float]:
    if df is None or len(df) < 15:
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    return round(float(close.iloc[-1]) - atr * atr_multiple, 4)


# ── 가상 포트폴리오 ────────────────────────────────────────────────────────────

class VirtualPortfolio:
    MAX_POSITIONS = 3
    POSITION_SIZE_PCT = 0.12   # 기본 10%
    STRONG_SIZE_PCT   = 0.15   # 강한 진입 12%

    def __init__(self, initial_capital: float) -> None:
        self.capital = initial_capital
        self.cash = initial_capital
        self.positions: List[BTPosition] = []
        self.trade_log: List[Dict[str, Any]] = []
        self.peak_value = initial_capital
        self._cooldown: Dict[str, date] = {}  # ticker → 재진입 금지 해제일
        self._monthly_entries: Dict[str, int] = {}  # ticker → 이번달 진입 횟수
        self._monthly_key: str = ""  # 현재 월 키 (YYYY-MM)
    @property
    def total_value(self) -> float:
        invested = sum(p.entry_price * p.shares for p in self.positions)
        return self.cash + invested

    def update_peak(self) -> None:
        self.peak_value = max(self.peak_value, self.total_value)

    def open_position(
        self,
        ticker: str,
        price: float,
        stop: float,
        entry_class: str,
        trade_date: date,
    ) -> bool:
        if len(self.positions) >= self.MAX_POSITIONS:
            return False
        if price <= 0:
            return False
        # 쿨다운 체크
        cooldown_end = self._cooldown.get(ticker)
        if cooldown_end is not None and trade_date < cooldown_end:
            return False

        # 월별 종목당 최대 3회 진입 제한
        month_key = trade_date.strftime("%Y-%m")
        if month_key != self._monthly_key:
            self._monthly_key = month_key
            self._monthly_entries = {}
        if self._monthly_entries.get(ticker, 0) >= 3:
            return False

        size_pct = (
            self.STRONG_SIZE_PCT
            if entry_class == EntryClass.STRONG_ENTRY.value
            else self.POSITION_SIZE_PCT
        )
        dollars = self.capital * size_pct
        if dollars > self.cash:
            dollars = self.cash * 0.95
        if dollars < 100:
            return False

        shares = round(dollars / price, 4)
        self.cash -= dollars
        self.positions.append(BTPosition(
            ticker=ticker,
            entry_price=price,
            stop_loss=stop,
            shares=shares,
            entry_date=trade_date,
            entry_class=entry_class,
            exposure_pct=size_pct * 100,
        ))
        self.trade_log.append({
            "date": trade_date, "action": "BUY",
            "ticker": ticker, "price": price,
            "shares": shares, "stop": stop,
            "entry_class": entry_class,
        })
        # 월별 진입 횟수 카운트
        month_key = trade_date.strftime("%Y-%m")
        self._monthly_entries[ticker] = self._monthly_entries.get(ticker, 0) + 1
        logger.debug(f"[BT] BUY {ticker} @ {price:.2f}  stop={stop:.2f}")
        return True

    def close_position(
        self,
        pos: BTPosition,
        price: float,
        trade_date: date,
        reason: str,
    ) -> float:
        proceeds = price * pos.shares
        self.cash += proceeds
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
        self.trade_log.append({
            "date": trade_date, "action": "SELL",
            "ticker": pos.ticker, "price": price,
            "shares": pos.shares, "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "hold_days": (trade_date - pos.entry_date).days,
        })
        # 큰 손실 발생 시 5일 쿨다운
        if pnl_pct <= -10.0:
            from datetime import timedelta
            self._cooldown[pos.ticker] = trade_date + timedelta(days=5)
            logger.debug(f"[BT] COOLDOWN {pos.ticker} until {self._cooldown[pos.ticker]}")
        logger.debug(f"[BT] SELL {pos.ticker} @ {price:.2f}  pnl={pnl_pct:.1f}%  reason={reason}")
        return pnl_pct

    def check_stops(
        self,
        sliced: Dict[str, pd.DataFrame],
        trade_date: date,
    ) -> List[str]:
        """손절가 hit 확인 → 청산."""
        exited: List[str] = []
        still_holding: List[BTPosition] = []
        for pos in self.positions:
            df = sliced.get(pos.ticker)
            if df is None or len(df) == 0:
                still_holding.append(pos)
                continue
            low_today = float(df["low"].iloc[-1])
            close_today = float(df["close"].iloc[-1])
            if low_today <= pos.stop_loss:
                exit_price = min(pos.stop_loss, close_today)
                self.close_position(pos, exit_price, trade_date, "stop_hit")
                exited.append(pos.ticker)
            else:
                still_holding.append(pos)
        self.positions = still_holding
        return exited

    def apply_signals(
        self,
        candidates: List[StockCandidate],
        sliced: Dict[str, pd.DataFrame],
        market_filter: MarketFilterResult,
        trade_date: date,
        atr_multiple: float = 1.5,
        rs_map: Dict[str, float] = None,
    ) -> List[str]:
        rs_map = rs_map or {}
        """
        FULL_EXIT 신호 종목 청산 → 빈 슬롯에 새 신호 진입.
        """
        exited: List[str] = []

        # 1) FULL_EXIT 신호 청산
        from core.models import HoldingStatus
        still_holding: List[BTPosition] = []
        for pos in self.positions:
            df = sliced.get(pos.ticker)
            if df is None or len(df) == 0:
                still_holding.append(pos)
                continue
            close = float(df["close"].iloc[-1])
            pnl_pct = (close - pos.entry_price) / pos.entry_price * 100
            hold_days = (trade_date - pos.entry_date).days

            if pnl_pct < -8.0:
                self.close_position(pos, close, trade_date, "structure_break")
                exited.append(pos.ticker)
            elif pnl_pct >= 6.5:
                self.close_position(pos, close, trade_date, "profit_target")
                exited.append(pos.ticker)
            elif hold_days >= (30 if pos.entry_class == EntryClass.STRONG_ENTRY.value else 20):
                self.close_position(pos, close, trade_date, "max_hold_days")
                exited.append(pos.ticker)
            else:
                still_holding.append(pos)
        self.positions = still_holding

        # 2) 새 신호 진입 (STRONG → GENERAL 순)
        entered: List[str] = []
        holding_tickers = {p.ticker for p in self.positions}

        priority = sorted(candidates, key=lambda c: (
            0 if c.entry_class == EntryClass.STRONG_ENTRY else
            1 if c.entry_class == EntryClass.GENERAL_ENTRY else 99
        ))

        for c in priority:
            if c.entry_class not in (EntryClass.STRONG_ENTRY, EntryClass.GENERAL_ENTRY):
                continue
            if c.ticker in holding_tickers:
                continue
            if len(self.positions) >= self.MAX_POSITIONS:
                break
            df = sliced.get(c.ticker)
            if df is None or len(df) < 15:
                continue
            price = float(df["close"].iloc[-1])
            stop = compute_stop_loss(df, atr_multiple=atr_multiple) or price * 0.92
            ok = self.open_position(
                ticker=c.ticker,
                price=price,
                stop=stop,
                entry_class=c.entry_class.value,
                trade_date=trade_date,
            )
            if ok:
                entered.append(c.ticker)
                holding_tickers.add(c.ticker)

        return exited + entered


# ── 메인 백테스트 엔진 ─────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(
        self,
        market_cfg: Dict[str, Any],
        stock_cfg: Dict[str, Any],
        risk_cfg: Dict[str, Any],
        initial_capital: float = 100_000.0,
    ) -> None:
        self.market_filter = MarketFilter(market_cfg)
        self.stock_engine = StockEngine(stock_cfg, risk_cfg)
        self.etf_engine = ETFEngine(market_cfg, risk_cfg)
        self.initial_capital = initial_capital
        self._atr_multiple = risk_cfg.get("stop_loss_atr_multiple", 1.5)

    def run(
        self,
        full_data: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
    ) -> BTSummary:
        trading_days = get_trading_days(full_data, start_date, end_date)
        if not trading_days:
            raise ValueError("거래일 없음 - 날짜 범위 또는 데이터 확인 필요")

        portfolio = VirtualPortfolio(self.initial_capital)
        daily_results: List[BTDayResult] = []
        prev_total = self.initial_capital
        weekly_pool: Optional[WeeklyStockPool] = None

        for trade_date in trading_days:
            sliced = slice_data(full_data, trade_date)

            qqq_df = sliced.get("QQQ")
            spy_df = sliced.get("SPY")
            vix_df = sliced.get("VIX")
            if qqq_df is None or spy_df is None or vix_df is None:
                continue

            mf = self.market_filter.run(qqq_df, spy_df, vix_df, trade_date)

            # 관망이면 손절 체크만
            if mf.pipeline_halt:
                exited = portfolio.check_stops(sliced, trade_date)
                total_val = portfolio.total_value
                daily_pnl = (total_val - prev_total) / prev_total * 100 if prev_total > 0 else 0.0
                daily_results.append(BTDayResult(
                    trade_date=trade_date,
                    market_state=mf.market_state.value,
                    action_mode=mf.action_mode.value,
                    new_entries=[],
                    exits=exited,
                    portfolio_value=total_val - portfolio.cash,
                    cash=portfolio.cash,
                    total_value=total_val,
                    daily_pnl_pct=daily_pnl,
                ))
                prev_total = total_val
                portfolio.update_peak()
                continue

            # 신호 생성
            account = AccountState(
                total_capital=self.initial_capital,
                daily_pnl_pct=0.0,
                weekly_pnl_pct=0.0,
                monthly_pnl_pct=0.0,
            )
            weekly_pool, candidates = self.stock_engine.run(
                universe_data=sliced,
                benchmark_df=qqq_df,
                market_filter=mf,
                account=account,
                analysis_date=trade_date,
                force_refresh=(trade_date.weekday() == 0),
            )

            # 손절 체크 → 신호 적용
            stop_exits = portfolio.check_stops(sliced, trade_date)
            rs_map = {e["ticker"]: e["rs_score"] for e in weekly_pool.pool} if weekly_pool else {}
            signal_changes = portfolio.apply_signals(candidates, sliced, mf, trade_date, atr_multiple=self._atr_multiple, rs_map=rs_map)

            total_val = portfolio.total_value
            daily_pnl = (total_val - prev_total) / prev_total * 100 if prev_total > 0 else 0.0
            daily_results.append(BTDayResult(
                trade_date=trade_date,
                market_state=mf.market_state.value,
                action_mode=mf.action_mode.value,
                new_entries=signal_changes,
                exits=stop_exits,
                portfolio_value=total_val - portfolio.cash,
                cash=portfolio.cash,
                total_value=total_val,
                daily_pnl_pct=daily_pnl,
            ))
            prev_total = total_val
            portfolio.update_peak()

        # ── 결과 집계 ────────────────────────────────────────────
        final_value = portfolio.total_value
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # 최대 낙폭
        peak = self.initial_capital
        max_dd = 0.0
        running = self.initial_capital
        for dr in daily_results:
            running = dr.total_value
            peak = max(peak, running)
            dd = (peak - running) / peak * 100
            max_dd = max(max_dd, dd)

        sells = [t for t in portfolio.trade_log if t["action"] == "SELL"]
        wins  = [t for t in sells if t.get("pnl_pct", 0) > 0]
        losses= [t for t in sells if t.get("pnl_pct", 0) <= 0]

        return BTSummary(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=round(total_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            total_trades=len(sells),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round(len(wins) / len(sells) * 100, 1) if sells else 0.0,
            avg_win_pct=round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0.0,
            avg_loss_pct=round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else 0.0,
            daily_results=daily_results,
            trade_log=portfolio.trade_log,
        )