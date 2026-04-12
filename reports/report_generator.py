from datetime import date
from typing import List, Dict, Optional
import pandas as pd
from core.models import (
    MarketFilterResult, MarketState,
    StockCandidate, ETFCandidate, EntryClass, ETFEntryClass,
    Position, HoldingStatus,
)


class ReportGenerator:

    def _current_price(self, df: Optional[pd.DataFrame]) -> Optional[float]:
        if df is None or len(df) == 0:
            return None
        return round(float(df["close"].iloc[-1]), 2)

    def _stop_loss(self, df: Optional[pd.DataFrame], atr_multiple: float = 2.0) -> Optional[float]:
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
        current = float(close.iloc[-1])
        return round(current - atr * atr_multiple, 2)

    def _position_size_pct(self, entry_class: EntryClass, market_filter: MarketFilterResult) -> str:
        cap = market_filter.exposure_cap
        if entry_class == EntryClass.STRONG_ENTRY:
            return f"{min(12, cap)}%"
        elif entry_class == EntryClass.GENERAL_ENTRY:
            return f"{min(10, cap)}%"
        elif entry_class == EntryClass.HOLD_CANDIDATE:
            return f"{min(5, cap)}%"
        return "0%"

    def _holding_label(self, status: HoldingStatus) -> str:
        mapping = {
            HoldingStatus.STRONG_HOLD: "강한 유지 ✅",
            HoldingStatus.HOLD: "유지",
            HoldingStatus.PARTIAL_REDUCE: "일부 축소 ⚠️",
            HoldingStatus.FULL_EXIT: "전량 정리 ❌",
        }
        return mapping.get(status, "유지")

    def generate(
        self,
        market_filter: MarketFilterResult,
        stock_candidates: List[StockCandidate],
        etf_candidates: List[ETFCandidate],
        analysis_date: date,
        universe_data: Dict[str, pd.DataFrame] = None,
        positions: List[Position] = None,
    ) -> str:
        universe_data = universe_data or {}
        positions = positions or []
        lines = []

        conclusion = self._conclusion(market_filter, stock_candidates, etf_candidates)
        lines.append("=" * 55)
        lines.append(f"  오늘 결론: {conclusion}")
        lines.append("=" * 55)

        lines.append("\n[시장 상태]")
        lines.append(f"  상태     : {market_filter.market_state.value}")
        lines.append(f"  행동지침 : {market_filter.action_mode.value}")
        lines.append(f"  QQQ      : {market_filter.qqq_signal}")
        lines.append(f"  SPY      : {market_filter.spy_signal}")
        lines.append(f"  VIX      : {market_filter.vix_signal}")
        lines.append(f"  해석     : {self._market_comment(market_filter)}")

        lines.append("\n[개별주 후보]")
        eligible = [c for c in stock_candidates if c.entry_class != EntryClass.INELIGIBLE]
        ineligible = [c for c in stock_candidates if c.entry_class == EntryClass.INELIGIBLE]
        if not eligible:
            lines.append("  오늘 진입 가능한 개별주 없음")
        for c in eligible:
            df = universe_data.get(c.ticker)
            price = self._current_price(df)
            stop = self._stop_loss(df)
            size = self._position_size_pct(c.entry_class, market_filter)
            lines.append(f"  ▶ {c.ticker} | {c.entry_class.name} | RS={c.rs_score:.2f}")
            lines.append(f"    매수가  : ${price}")
            lines.append(f"    손절가  : ${stop}")
            lines.append(f"    권장비중: {size}")
        if ineligible:
            lines.append("  [탈락 종목]")
            for c in ineligible:
                lines.append(f"  ✕ {c.ticker} | Gate{c.gate_result.failure_gate} | {c.gate_result.failure_reason}")

        lines.append("\n[ETF 후보]")
        etf_approved = [e for e in etf_candidates if e.entry_class == ETFEntryClass.APPROVED]
        etf_rejected = [e for e in etf_candidates if e.entry_class == ETFEntryClass.INELIGIBLE]
        if not etf_approved:
            lines.append("  오늘 진입 가능한 ETF 없음")
        for e in etf_approved:
            df = universe_data.get(e.ticker)
            price = self._current_price(df)
            stop = self._stop_loss(df)
            lines.append(f"  ▶ {e.ticker} | APPROVED | 추세={e.gate_result.gate2_trend.value}")
            lines.append(f"    매수가  : ${price}")
            lines.append(f"    손절가  : ${stop}")
            lines.append(f"    권장비중: {min(10, market_filter.exposure_cap)}%")
        if etf_rejected:
            lines.append("  [탈락 ETF]")
            for e in etf_rejected:
                lines.append(f"  ✕ {e.ticker} | Gate{e.gate_result.failure_gate} | {e.gate_result.failure_reason}")

        lines.append("\n[보유 종목 관리]")
        if not positions:
            lines.append("  현재 보유 종목 없음")
        for p in positions:
            label = self._holding_label(p.holding_status)
            pnl = (p.current_price - p.entry_price) / p.entry_price * 100
            pnl_str = f"+{pnl:.1f}%" if pnl >= 0 else f"{pnl:.1f}%"
            lines.append(f"  {p.ticker} | {label} | 수익: {pnl_str} | RS={p.rs_score:.2f}")

        lines.append("\n[리스크 상태]")
        if market_filter.pipeline_halt:
            lines.append("  ❌ 신규 진입 금지 (시장 관망)")
        else:
            lines.append("  ✅ 신규 진입 가능")

        lines.append("\n[실행 요약]")
        lines.append(f"  개별주 진입 후보: {len(eligible)}건")
        lines.append(f"  ETF 진입 후보  : {len(etf_approved)}건")
        lines.append(f"  보유 종목 수   : {len(positions)}개")
        lines.append(f"  최대 노출 한도 : {market_filter.exposure_cap}%")
        lines.append("=" * 55)

        return "\n".join(lines)

    def _conclusion(self, mf, stocks, etfs) -> str:
        if mf.pipeline_halt:
            return "매매 불가"
        if any(c.entry_class == EntryClass.STRONG_ENTRY for c in stocks):
            return "강한 매매 가능"
        approved_etfs = [e for e in etfs if e.entry_class == ETFEntryClass.APPROVED]
        if approved_etfs or any(c.entry_class == EntryClass.GENERAL_ENTRY for c in stocks):
            return "매매 가능"
        if any(c.entry_class == EntryClass.HOLD_CANDIDATE for c in stocks):
            return "보류"
        return "매매 불가"

    def _market_comment(self, mf: MarketFilterResult) -> str:
        if mf.market_state == MarketState.GOOD:
            return "시장 추세 양호. 공격적 신규 진입 가능."
        elif mf.market_state == MarketState.NEUTRAL:
            return "시장 흐름 혼조. 선별적 진입만 허용."
        elif mf.market_state == MarketState.BAD:
            if mf.pipeline_halt:
                return "시장 추세 약화 + VIX 급등. 신규 진입 금지."
            return "시장 추세 약화. 현금 비중 우선."
        return "판단 불가."