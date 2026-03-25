"""
portfolio/portfolio_manager.py  (v1.6)

Fix 1: sync_rs_scores(rs_map) 추가.
  - 파이프라인 Step 3(StockEngine) 직후 호출.
  - 포트폴리오 보유 종목의 rs_score를 당일 StockEngine 결과로 갱신.
  - ReplacementEvaluator가 실행되기 전에 반드시 호출되어야 함.

Fix 2: ExecutedEntry 히스토리 영속화.
  - confirm_executions() 호출 시 _executed_today 리스트에 추가.
  - save_state()에서 executed_entries 직렬화.
  - _load_state()에서 당일분만 복원 (익일에는 초기화).
  - executed_tickers_today 프로퍼티: PriorityEngine에 전달해 same-day 재제안 차단.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from core.models import (
    AccountState,
    DailyEntryCounter,
    EntryClass,
    ExecutedEntry,
    HoldingStatus,
    Position,
    PortfolioState,
)

logger = logging.getLogger(__name__)


def _evaluate_holding_status(
    position: Position,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> HoldingStatus:
    if len(df) == 0:
        return position.holding_status
    current_price = float(df["close"].iloc[-1])
    pnl_pct = (
        (current_price - position.entry_price) / position.entry_price * 100
        if position.entry_price > 0 else 0.0
    )
    structure_break: float = cfg.get("structure_break_pct", -8.0)
    trend_weak: float = cfg.get("trend_weak_pct", -4.0)
    strong_hold_rs: float = cfg.get("strong_hold_rs_threshold", 5.0)

    if pnl_pct < structure_break:
        return HoldingStatus.FULL_EXIT
    if pnl_pct < trend_weak:
        return HoldingStatus.PARTIAL_REDUCE
    if position.rs_score >= strong_hold_rs and pnl_pct >= 0:
        return HoldingStatus.STRONG_HOLD
    return HoldingStatus.HOLD


def _serialize_position(pos: Position) -> Dict[str, Any]:
    return {
        "ticker": pos.ticker,
        "entry_price": pos.entry_price,
        "current_price": pos.current_price,
        "quantity": pos.quantity,
        "entry_date": pos.entry_date.isoformat(),
        "holding_status": pos.holding_status.value,
        "rs_score": pos.rs_score,
    }


def _deserialize_position(d: Dict[str, Any]) -> Position:
    return Position(
        ticker=d["ticker"],
        entry_price=float(d["entry_price"]),
        current_price=float(d["current_price"]),
        quantity=float(d["quantity"]),
        entry_date=date.fromisoformat(d["entry_date"]),
        holding_status=HoldingStatus(d.get("holding_status", HoldingStatus.HOLD.value)),
        rs_score=float(d.get("rs_score", 0.0)),
    )


def _serialize_executed_entry(e: ExecutedEntry) -> Dict[str, Any]:
    return {
        "ticker": e.ticker,
        "entry_class": e.entry_class,
        "shares_executed": e.shares_executed,
        "execution_price": e.execution_price,
        "execution_date": e.execution_date.isoformat(),
        "is_replacement": e.is_replacement,
        "outgoing_ticker": e.outgoing_ticker,
    }


def _deserialize_executed_entry(d: Dict[str, Any]) -> ExecutedEntry:
    return ExecutedEntry(
        ticker=d["ticker"],
        entry_class=d["entry_class"],
        shares_executed=float(d["shares_executed"]),
        execution_price=float(d["execution_price"]),
        execution_date=date.fromisoformat(d["execution_date"]),
        is_replacement=bool(d.get("is_replacement", False)),
        outgoing_ticker=d.get("outgoing_ticker"),
    )


def _merge_or_add_position(
    positions: List[Position],
    ticker: str,
    shares: float,
    price: float,
    entry_date: date,
) -> None:
    existing = next((p for p in positions if p.ticker == ticker), None)
    if existing is not None:
        old_qty = existing.quantity
        new_qty = old_qty + shares
        if new_qty > 0:
            existing.entry_price = round(
                (existing.entry_price * old_qty + price * shares) / new_qty, 4
            )
        existing.quantity = new_qty
        existing.current_price = price
        logger.info(
            f"[PortfolioManager] Merged {ticker}: qty {old_qty:.4f} → {new_qty:.4f}, "
            f"avg_entry=${existing.entry_price:.2f}"
        )
    else:
        positions.append(Position(
            ticker=ticker,
            entry_price=price,
            current_price=price,
            quantity=shares,
            entry_date=entry_date,
        ))
        logger.info(
            f"[PortfolioManager] New position {ticker}: {shares:.4f}sh @ ${price:.2f}"
        )


class PortfolioManager:
    def __init__(
        self,
        state_file: Optional[str] = None,
        portfolio_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._state_file: Optional[Path] = Path(state_file) if state_file else None
        self._cfg: Dict[str, Any] = portfolio_cfg or {}
        self._portfolio: PortfolioState = PortfolioState()
        self._account: AccountState = AccountState(total_capital=0.0)
        self._counter: DailyEntryCounter = DailyEntryCounter()
        self._executed_today: List[ExecutedEntry] = []

    @property
    def portfolio(self) -> PortfolioState:
        return self._portfolio

    @property
    def account(self) -> AccountState:
        return self._account

    @property
    def daily_counter(self) -> DailyEntryCounter:
        return self._counter

    @property
    def executed_tickers_today(self) -> Set[str]:
        """
        Set of tickers confirmed as executed on today's date.
        Passed to PriorityEngine.run() to prevent same-day re-proposal
        of already-executed tickers in new_entries or replacement incoming.
        """
        return {e.ticker for e in self._executed_today}

    def initialize(self, total_capital: float, today: date) -> None:
        if self._state_file and self._state_file.exists():
            self._load_state(total_capital, today)
        else:
            self._portfolio = PortfolioState(total_capital=total_capital, cash=total_capital)
            self._account = AccountState(total_capital=total_capital)
            logger.info(f"[PortfolioManager] Fresh state, capital=${total_capital:,.0f}")

        if self._counter.analysis_date != today:
            old = self._counter.analysis_date
            self._counter = DailyEntryCounter(analysis_date=today)
            self._executed_today = []
            logger.info(f"[PortfolioManager] Counter + executed history reset: {old} → {today}")

    def sync_rs_scores(self, rs_map: Dict[str, float]) -> None:
        """
        Update portfolio positions' rs_score from today's StockEngine output.
        Call this AFTER StockEngine.run() and BEFORE PriorityEngine.run().
        """
        updated = 0
        for pos in self._portfolio.positions:
            if pos.ticker in rs_map:
                old_score = pos.rs_score
                pos.rs_score = rs_map[pos.ticker]
                if old_score != pos.rs_score:
                    logger.info(
                        f"[PortfolioManager] rs_score synced: "
                        f"{pos.ticker} {old_score:.2f} → {pos.rs_score:.2f}"
                    )
                    updated += 1

        logger.info(
            f"[PortfolioManager] sync_rs_scores: "
            f"{updated} positions updated, "
            f"{len(self._portfolio.positions) - updated} unchanged (not in today's pool)"
        )

    def update_positions(self, price_data: Dict[str, pd.DataFrame]) -> None:
        for pos in self._portfolio.positions:
            df = price_data.get(pos.ticker)
            if df is not None and len(df) > 0:
                pos.current_price = float(df["close"].iloc[-1])
                pos.holding_status = _evaluate_holding_status(pos, df, self._cfg)

    def confirm_executions(self, executed: List[ExecutedEntry]) -> None:
        """
        Record broker-confirmed trades. The ONLY path that updates DailyEntryCounter.
        Also appends each ExecutedEntry to _executed_today.
        """
        for entry in executed:
            if entry.is_replacement:
                if entry.outgoing_ticker:
                    for pos in self._portfolio.positions:
                        if pos.ticker == entry.outgoing_ticker:
                            pos.quantity = round(pos.quantity * 0.5, 4)
                            logger.info(
                                f"[PortfolioManager] Replacement: "
                                f"{pos.ticker} reduced to {pos.quantity:.4f}sh (50%)"
                            )
                            break

                _merge_or_add_position(
                    self._portfolio.positions,
                    ticker=entry.ticker,
                    shares=entry.shares_executed,
                    price=entry.execution_price,
                    entry_date=entry.execution_date,
                )
                self._counter.replacements_today += 1

            else:
                _merge_or_add_position(
                    self._portfolio.positions,
                    ticker=entry.ticker,
                    shares=entry.shares_executed,
                    price=entry.execution_price,
                    entry_date=entry.execution_date,
                )
                if entry.entry_class == EntryClass.STRONG_ENTRY.value:
                    self._counter.strong_entries_today += 1
                else:
                    self._counter.general_entries_today += 1

            self._executed_today.append(entry)

        logger.info(
            f"[PortfolioManager] Counter: "
            f"strong={self._counter.strong_entries_today}, "
            f"general={self._counter.general_entries_today}, "
            f"replacements={self._counter.replacements_today}. "
            f"Executed today: {list(self.executed_tickers_today)}"
        )

    def save_state(self) -> None:
        if self._state_file is None:
            return
        try:
            data = {
                "account": {
                    "total_capital": self._account.total_capital,
                    "daily_pnl": self._account.daily_pnl,
                    "weekly_pnl": self._account.weekly_pnl,
                    "monthly_pnl": self._account.monthly_pnl,
                    "daily_pnl_pct": self._account.daily_pnl_pct,
                    "weekly_pnl_pct": self._account.weekly_pnl_pct,
                    "monthly_pnl_pct": self._account.monthly_pnl_pct,
                },
                "positions": [_serialize_position(p) for p in self._portfolio.positions],
                "daily_counter": {
                    "strong_entries_today": self._counter.strong_entries_today,
                    "general_entries_today": self._counter.general_entries_today,
                    "replacements_today": self._counter.replacements_today,
                    "analysis_date": (
                        self._counter.analysis_date.isoformat()
                        if self._counter.analysis_date else None
                    ),
                },
                "executed_entries": [
                    _serialize_executed_entry(e) for e in self._executed_today
                ],
                "saved_at": datetime.now().isoformat(),
            }
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"[PortfolioManager] Saved: {len(self._portfolio.positions)} positions, "
                f"{len(self._executed_today)} executed entries"
            )
        except Exception as exc:
            logger.error(f"[PortfolioManager] Save failed: {exc}")

    def _load_state(self, fallback_capital: float, today: date) -> None:
        try:
            with open(self._state_file, "r") as f:  # type: ignore[arg-type]
                data = json.load(f)

            acct = data.get("account", {})
            self._account = AccountState(
                total_capital=acct.get("total_capital", fallback_capital),
                daily_pnl=acct.get("daily_pnl", 0.0),
                weekly_pnl=acct.get("weekly_pnl", 0.0),
                monthly_pnl=acct.get("monthly_pnl", 0.0),
                daily_pnl_pct=acct.get("daily_pnl_pct", 0.0),
                weekly_pnl_pct=acct.get("weekly_pnl_pct", 0.0),
                monthly_pnl_pct=acct.get("monthly_pnl_pct", 0.0),
            )

            positions = []
            for p_data in data.get("positions", []):
                try:
                    positions.append(_deserialize_position(p_data))
                except Exception as exc:
                    logger.warning(f"[PortfolioManager] Bad position skipped: {exc}")

            self._portfolio = PortfolioState(
                positions=positions,
                total_capital=self._account.total_capital,
                cash=self._account.total_capital,
            )

            ctr = data.get("daily_counter", {})
            saved_date_str = ctr.get("analysis_date")
            saved_date = date.fromisoformat(saved_date_str) if saved_date_str else None

            if saved_date == today:
                self._counter = DailyEntryCounter(
                    strong_entries_today=ctr.get("strong_entries_today", 0),
                    general_entries_today=ctr.get("general_entries_today", 0),
                    replacements_today=ctr.get("replacements_today", 0),
                    analysis_date=saved_date,
                )
                self._executed_today = []
                for ed in data.get("executed_entries", []):
                    try:
                        entry = _deserialize_executed_entry(ed)
                        if entry.execution_date == today:
                            self._executed_today.append(entry)
                    except Exception as exc:
                        logger.warning(f"[PortfolioManager] Bad executed_entry skipped: {exc}")
                logger.info(
                    f"[PortfolioManager] Restored counter + {len(self._executed_today)} "
                    f"executed entries for {today}"
                )
            else:
                self._counter = DailyEntryCounter(analysis_date=saved_date)
                self._executed_today = []

            logger.info(
                f"[PortfolioManager] Loaded: {len(positions)} positions, "
                f"capital=${self._account.total_capital:,.0f}"
            )
        except Exception as exc:
            logger.error(f"[PortfolioManager] Load failed: {exc}. Using defaults.")
            self._account = AccountState(total_capital=fallback_capital)
            self._portfolio = PortfolioState(
                total_capital=fallback_capital, cash=fallback_capital
            )
            self._counter = DailyEntryCounter()
            self._executed_today = []