# reports/report_generator.py  (v1.2)
# 가독성 개선: 한글화, 섹션 정리, 탈락 요약 압축

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from core.models import (
    ETFCandidate,
    ETFEntryClass,
    EntryClass,
    HoldingStatus,
    MarketFilterResult,
    MarketState,
    Position,
    StockCandidate,
)


class ReportGenerator:

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _current_price(self, df: Optional[pd.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "N/A"
        return f"${float(df['close'].iloc[-1]):.2f}"

    def _stop_loss(self, df: Optional[pd.DataFrame], atr_multiple: float = 2.0) -> str:
        if df is None or len(df) < 15:
            return "N/A"
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        stop = float(close.iloc[-1]) - atr * atr_multiple
        return f"${stop:.2f}"

    def _position_size_pct(
        self, entry_class: EntryClass, market_filter: MarketFilterResult
    ) -> str:
        cap = market_filter.exposure_cap
        if entry_class == EntryClass.STRONG_ENTRY:
            return f"{min(12, cap)}%"
        elif entry_class == EntryClass.GENERAL_ENTRY:
            return f"{min(10, cap)}%"
        elif entry_class == EntryClass.HOLD_CANDIDATE:
            return f"{min(5, cap)}%"
        return "0%"

    def _holding_label(self, status: HoldingStatus) -> str:
        return {
            HoldingStatus.STRONG_HOLD:    "강한 유지 ✅",
            HoldingStatus.HOLD:           "유지",
            HoldingStatus.PARTIAL_REDUCE: "일부 축소 ⚠️",
            HoldingStatus.FULL_EXIT:      "전량 정리 ❌",
        }.get(status, "유지")

    def _entry_label(self, entry_class: EntryClass) -> str:
        return {
            EntryClass.STRONG_ENTRY:   "★ 강한 진입",
            EntryClass.GENERAL_ENTRY:  "▶ 일반 진입",
            EntryClass.HOLD_CANDIDATE: "△ 보류",
            EntryClass.INELIGIBLE:     "✕ 탈락",
        }.get(entry_class, "✕ 탈락")

    def _gate_summary(self, c: StockCandidate) -> str:
        g = c.gate_result
        def _g1(v):
            from core.models import MarketStateGate
            return {"PASS": "✅", "EXCEPTION_PASS": "⚡", "FAIL": "❌"}.get(v.value, v.value)
        def _g2(v):
            from core.models import RelativeStrengthGate
            return {"TOP_TIER_PASS": "✅✅", "PASS": "✅", "FAIL": "❌"}.get(v.value, v.value)
        def _g3(v):
            from core.models import PullbackStructureGate
            return {"PREMIUM_PASS": "✅✅", "PASS": "✅", "FAIL": "❌"}.get(v.value, v.value)
        def _g4(v):
            from core.models import ReStrengtheningGate
            return {"PREMIUM_PASS": "✅✅", "PASS": "✅", "FAIL": "❌"}.get(v.value, v.value)
        def _g5(v):
            from core.models import RiskStatusGate
            return {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(v.value, v.value)
        return (
            f"시장{_g1(g.gate1_market)} "
            f"RS{_g2(g.gate2_rs)} "
            f"눌림{_g3(g.gate3_pullback)} "
            f"재강세{_g4(g.gate4_restrength)} "
            f"리스크{_g5(g.gate5_risk)}"
        )

    def _market_comment(self, mf: MarketFilterResult) -> str:
        if mf.market_state == MarketState.GOOD:
            return "QQQ·SPY 추세 양호, VIX 안정 → 공격적 신규 진입 가능."
        elif mf.market_state == MarketState.NEUTRAL:
            return "시장 흐름 혼조 → 선별적 진입만 허용, 비중 축소."
        elif mf.market_state == MarketState.BAD:
            if mf.pipeline_halt:
                return "추세 약화 + VIX 급등 → 신규 진입 전면 금지."
            return "추세 약화 → 현금 비중 우선, 매우 엄격하게 선별."
        return "판단 불가."

    def _conclusion(
        self,
        mf: MarketFilterResult,
        stocks: List[StockCandidate],
        etfs: List[ETFCandidate],
    ) -> str:
        if mf.pipeline_halt:
            return "매매 불가 🚫"
        if any(c.entry_class == EntryClass.STRONG_ENTRY for c in stocks):
            return "강한 매매 가능 🔥"
        approved_etfs = [e for e in etfs if e.entry_class == ETFEntryClass.APPROVED]
        if approved_etfs or any(c.entry_class == EntryClass.GENERAL_ENTRY for c in stocks):
            return "매매 가능 ✅"
        if any(c.entry_class == EntryClass.HOLD_CANDIDATE for c in stocks):
            return "보류 👀"
        return "매매 불가 🚫"

    # ── 메인 generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        market_filter: MarketFilterResult,
        stock_candidates: List[StockCandidate],
        etf_candidates: List[ETFCandidate],
        analysis_date: date,
        universe_data: Optional[Dict[str, pd.DataFrame]] = None,
        positions: Optional[List[Position]] = None,
    ) -> str:
        universe_data = universe_data or {}
        positions = positions or []
        L = []

        conclusion = self._conclusion(market_filter, stock_candidates, etf_candidates)

        L.append("=" * 60)
        L.append(f"  스윙 트레이딩 시스템  |  {analysis_date}")
        L.append("=" * 60)
        L.append(f"  오늘 결론: {conclusion}")
        L.append("=" * 60)

        # ── 시장 상태 ────────────────────────────────────────────
        L.append("\n━━ 시장 상태 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        state_kr = {
            "GOOD": "좋음 🟢", "NEUTRAL": "중립 🟡", "BAD": "나쁨 🔴"
        }.get(market_filter.market_state.value, market_filter.market_state.value)
        action_kr = {
            "AGGRESSIVE": "공격",
            "LIMITED_AGGRESSIVE": "제한적 공격",
            "DEFENSIVE": "보수",
            "STAY_OUT": "관망",
        }.get(market_filter.action_mode.value, market_filter.action_mode.value)
        L.append(f"  상태     : {state_kr}")
        L.append(f"  행동지침 : {action_kr}  (최대노출 {market_filter.exposure_cap}%)")
        L.append(f"  QQQ      : {market_filter.qqq_signal}")
        L.append(f"  SPY      : {market_filter.spy_signal}")
        L.append(f"  VIX      : {market_filter.vix_signal}")
        L.append(f"  해석     : {self._market_comment(market_filter)}")

        # ── 주간 강종목 풀 ───────────────────────────────────────
        L.append("\n━━ 주간 강종목 풀 (상위 10) ━━━━━━━━━━━━━━━━━━━━━━━━")
        sorted_pool = sorted(stock_candidates, key=lambda x: x.pool_rank)
        for c in sorted_pool:
            L.append(f"  #{c.pool_rank:>2}  {c.ticker:<6}  RS={c.rs_score:+.2f}")

        # ── 개별주 진입 후보 ─────────────────────────────────────
        L.append("\n━━ 개별주 진입 후보 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        eligible   = [c for c in stock_candidates if c.entry_class != EntryClass.INELIGIBLE]
        ineligible = [c for c in stock_candidates if c.entry_class == EntryClass.INELIGIBLE]

        if not eligible:
            L.append("  진입 가능한 개별주 없음")
        else:
            for c in sorted(eligible, key=lambda x: (
                0 if x.entry_class == EntryClass.STRONG_ENTRY else
                1 if x.entry_class == EntryClass.GENERAL_ENTRY else 2
            )):
                df = universe_data.get(c.ticker)
                L.append(f"\n  {self._entry_label(c.entry_class)}  {c.ticker}  (RS={c.rs_score:+.2f}  풀순위#{c.pool_rank})")
                L.append(f"  게이트  : {self._gate_summary(c)}")
                L.append(f"  매수가  : {self._current_price(df)}")
                L.append(f"  손절가  : {self._stop_loss(df)}")
                L.append(f"  권장비중: {self._position_size_pct(c.entry_class, market_filter)}")
                if c.human_review_flags.pullback_quality_needed:
                    L.append(f"  👁 눌림  : {c.human_review_flags.pullback_notes[:80]}")
                if c.human_review_flags.restrengthening_quality_needed:
                    L.append(f"  👁 재강세: {c.human_review_flags.restrengthening_notes[:80]}")

        if ineligible:
            L.append(f"\n  [탈락 {len(ineligible)}종목]  " + "  ".join(
                f"{c.ticker}(G{c.gate_result.failure_gate})" for c in ineligible
            ))

        # ── ETF 후보 ─────────────────────────────────────────────
        L.append("\n━━ ETF 후보 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        approved = [e for e in etf_candidates if e.entry_class == ETFEntryClass.APPROVED]
        rejected = [e for e in etf_candidates if e.entry_class == ETFEntryClass.INELIGIBLE]

        if not approved:
            L.append("  진입 가능한 ETF 없음")
        else:
            for e in approved:
                df = universe_data.get(e.ticker)
                trend_kr = {"PREMIUM_PASS": "✅✅ 우수", "PASS": "✅ 양호", "FAIL": "❌"}.get(
                    e.gate_result.gate2_trend.value, e.gate_result.gate2_trend.value
                )
                L.append(f"\n  ▶ ETF 진입  {e.ticker}")
                L.append(f"  추세    : {trend_kr}")
                L.append(f"  매수가  : {self._current_price(df)}")
                L.append(f"  손절가  : {self._stop_loss(df)}")
                L.append(f"  권장비중: {min(10, market_filter.exposure_cap)}%")

        if rejected:
            L.append(f"\n  [탈락 ETF]  " + "  ".join(
                f"{e.ticker}(G{e.gate_result.failure_gate})" for e in rejected
            ))

        # ── 보유 종목 관리 ───────────────────────────────────────
        L.append("\n━━ 보유 종목 관리 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if not positions:
            L.append("  현재 보유 종목 없음")
        else:
            for p in positions:
                label = self._holding_label(p.holding_status)
                pnl = (
                    (p.current_price - p.entry_price) / p.entry_price * 100
                    if p.entry_price > 0 else 0.0
                )
                pnl_str = f"+{pnl:.1f}%" if pnl >= 0 else f"{pnl:.1f}%"
                L.append(
                    f"  {p.ticker:<6}  {label:<14}  수익: {pnl_str:>7}  RS={p.rs_score:+.2f}"
                )

        # ── 리스크 상태 ──────────────────────────────────────────
        L.append("\n━━ 리스크 상태 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if market_filter.pipeline_halt:
            L.append(f"  ❌ 신규 진입 금지  ({market_filter.halt_reason})")
        else:
            L.append("  ✅ 신규 진입 가능")

        # ── 실행 요약 ─────────────────────────────────────────────
        strong_cnt  = sum(1 for c in stock_candidates if c.entry_class == EntryClass.STRONG_ENTRY)
        general_cnt = sum(1 for c in stock_candidates if c.entry_class == EntryClass.GENERAL_ENTRY)
        hold_cnt    = sum(1 for c in stock_candidates if c.entry_class == EntryClass.HOLD_CANDIDATE)

        L.append("\n━━ 실행 요약 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        L.append(f"  강한 진입  : {strong_cnt}건")
        L.append(f"  일반 진입  : {general_cnt}건")
        L.append(f"  보류       : {hold_cnt}건")
        L.append(f"  ETF 진입   : {len(approved)}건")
        L.append(f"  보유 종목  : {len(positions)}개")
        L.append(f"  최대 노출  : {market_filter.exposure_cap}%")
        L.append("=" * 60)

        return "\n".join(L)