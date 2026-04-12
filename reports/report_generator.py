from datetime import date
from typing import List
from core.models import (
    MarketFilterResult, MarketState, ActionMode,
    StockCandidate, ETFCandidate, EntryClass, ETFEntryClass,
)


class ReportGenerator:

    def generate(
        self,
        market_filter: MarketFilterResult,
        stock_candidates: List[StockCandidate],
        etf_candidates: List[ETFCandidate],
        analysis_date: date,
    ) -> str:
        lines = []

        # 1. 한 줄 결론
        conclusion = self._conclusion(market_filter, stock_candidates, etf_candidates)
        lines.append("=" * 55)
        lines.append(f"  오늘 결론: {conclusion}")
        lines.append("=" * 55)

        # 2. 시장 필터
        lines.append("\n[시장 상태]")
        lines.append(f"  상태     : {market_filter.market_state.value}")
        lines.append(f"  행동지침 : {market_filter.action_mode.value}")
        lines.append(f"  QQQ      : {market_filter.qqq_signal}")
        lines.append(f"  SPY      : {market_filter.spy_signal}")
        lines.append(f"  VIX      : {market_filter.vix_signal}")
        lines.append(f"  해석     : {self._market_comment(market_filter)}")

        # 3. 개별주 후보
        lines.append("\n[개별주 후보]")
        eligible = [c for c in stock_candidates if c.entry_class != EntryClass.INELIGIBLE]
        if not eligible:
            lines.append("  오늘 진입 가능한 개별주 없음")
        for c in eligible:
            lines.append(f"  {c.ticker} | {c.entry_class.name} | RS={c.rs_score:.2f}")
            lines.append(f"    탈락사유: 없음 (통과)")

        # 4. ETF 후보
        lines.append("\n[ETF 후보]")
        etf_approved = [e for e in etf_candidates if e.entry_class == ETFEntryClass.APPROVED]
        if not etf_approved:
            lines.append("  오늘 진입 가능한 ETF 없음")
        for e in etf_approved:
            lines.append(f"  {e.ticker} | APPROVED | 추세={e.gate_result.gate2_trend.value}")

        # 5. 리스크 상태
        lines.append("\n[리스크 상태]")
        if market_filter.pipeline_halt:
            lines.append("  ❌ 신규 진입 금지 (시장 관망)")
        else:
            lines.append("  ✅ 신규 진입 가능")

        # 6. 실행 요약
        lines.append("\n[실행 요약]")
        lines.append(f"  개별주 진입 후보: {len(eligible)}건")
        lines.append(f"  ETF 진입 후보  : {len(etf_approved)}건")
        lines.append(f"  최대 노출 한도 : {market_filter.exposure_cap}%")
        lines.append("=" * 55)

        return "\n".join(lines)

    def _conclusion(self, mf, stocks, etfs) -> str:
        if mf.pipeline_halt:
            return "매매 불가"
        eligible_stocks = [c for c in stocks if c.entry_class == EntryClass.STRONG_ENTRY]
        approved_etfs = [e for e in etfs if e.entry_class == ETFEntryClass.APPROVED]
        if eligible_stocks:
            return "강한 매매 가능"
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