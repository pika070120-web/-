"""
priority/priority_engine.py  (v1.6)

Fix 2: executed_tickers_today 파라미터 추가.
  - PriorityEngine.run(): executed ticker는 new_entries 후보에서 제외.
  - ReplacementEvaluator.evaluate(): executed ticker는 replacement incoming 후보에서 제외.
  - 당일 차단만 적용. 익일 파이프라인에는 영향 없음.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from core.constants import (
    MAX_GENERAL_ENTRIES_PER_DAY,
    MAX_HOLDINGS,
    MAX_REPLACEMENTS_PER_DAY,
    MAX_STRONG_ENTRIES_PER_DAY,
    REPLACEMENT_REDUCTION_RATIO,
)
from core.models import (
    ActionMode,
    DailyEntryCounter,
    EntryClass,
    ETFEntryClass,
    HoldingStatus,
    MarketFilterResult,
    PortfolioState,
    RankedActionList,
    RankedCandidate,
    ReplacementAction,
    ReStrengtheningGate,
    StockCandidate,
    ETFCandidate,
)

logger = logging.getLogger(__name__)


class CandidateRanker:
    """Builds a single ranked list from stock + ETF candidates."""

    def rank(
        self,
        stocks: List[StockCandidate],
        etfs: List[ETFCandidate],
        action_mode: ActionMode,
    ) -> List[RankedCandidate]:
        strong = sorted(
            [c for c in stocks if c.entry_class == EntryClass.STRONG_ENTRY],
            key=lambda x: x.rs_score, reverse=True,
        )
        general = sorted(
            [c for c in stocks if c.entry_class == EntryClass.GENERAL_ENTRY],
            key=lambda x: x.rs_score, reverse=True,
        )
        etf_approved = sorted(
            [c for c in etfs if c.entry_class == ETFEntryClass.APPROVED],
            key=lambda x: x.trend_score, reverse=True,
        )

        def s_to_r(c: StockCandidate) -> RankedCandidate:
            return RankedCandidate(
                0, c.ticker, "STOCK", c.entry_class.value,
                c.rs_score, c.human_review_flags,
            )

        def e_to_r(c: ETFCandidate) -> RankedCandidate:
            return RankedCandidate(
                0, c.ticker, "ETF", ETFEntryClass.APPROVED.value,
                c.trend_score, None,
            )

        if action_mode == ActionMode.LIMITED_AGGRESSIVE:
            ordered = (
                [s_to_r(c) for c in strong]
                + [e_to_r(c) for c in etf_approved]
                + [s_to_r(c) for c in general]
            )
        else:
            ordered = (
                [s_to_r(c) for c in strong]
                + [s_to_r(c) for c in general]
                + [e_to_r(c) for c in etf_approved]
            )

        for i, c in enumerate(ordered):
            c.rank = i + 1
        return ordered


class ReplacementEvaluator:
    """
    Evaluates replacement conditions. All 3 conditions required (Option B):
      1. Existing position weakening
      2. New candidate RS gap >= superiority_threshold
      3. New candidate gate4 == PREMIUM_PASS

    Slot guard: current + planned_new + 1 ≤ MAX_HOLDINGS (outgoing stays at 50%).

    Duplicate ticker guards:
      - current_tickers: already in portfolio
      - planned_new_entry_tickers: selected as same-day new_entry (Fix 1 from v1.5)
      - executed_tickers_today: already confirmed executed today (Fix 2 from v1.6)

    Pre-risk count conservative behavior (Fix 2 from v1.4):
      planned_new_entries uses pre-risk count intentionally.
      Some entries may later be blocked by RiskEngine, but at this stage
      we cannot know — conservative approach prevents MAX_HOLDINGS overflow
      in the worst case. This is by design.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.weakening_threshold: float = cfg.get("weakening_threshold", -3.0)
        self.superiority_threshold: float = cfg.get("superiority_threshold", 3.0)

    def _is_weakening(self, position) -> bool:
        return (
            position.holding_status in (HoldingStatus.PARTIAL_REDUCE, HoldingStatus.FULL_EXIT)
            or position.rs_score < self.weakening_threshold
        )

    def evaluate(
        self,
        stock_candidates: List[StockCandidate],
        portfolio: PortfolioState,
        counter: DailyEntryCounter,
        planned_new_entries: int = 0,
        planned_new_entry_tickers: Optional[Set[str]] = None,
        executed_tickers_today: Optional[Set[str]] = None,
    ) -> Optional[ReplacementAction]:
        """
        planned_new_entries: pre-risk count (conservative by design — see class docstring).
        planned_new_entry_tickers: same-day new_entry tickers (no ticker overlap).
        executed_tickers_today: already executed today — excluded from incoming (Fix 2).
        """
        _planned_tickers: Set[str] = planned_new_entry_tickers or set()
        _executed: Set[str] = executed_tickers_today or set()

        if counter.replacements_today >= MAX_REPLACEMENTS_PER_DAY:
            logger.debug("[ReplacementEvaluator] Daily replacement limit reached")
            return None

        total_after = portfolio.num_holdings + planned_new_entries + 1
        if total_after > MAX_HOLDINGS:
            logger.info(
                f"[ReplacementEvaluator] Replacement blocked: "
                f"current({portfolio.num_holdings}) + "
                f"planned_new(pre-risk={planned_new_entries}) + "
                f"incoming(1) = {total_after} > MAX_HOLDINGS({MAX_HOLDINGS}). "
                f"[Intentionally conservative: pre-risk count used.]"
            )
            return None

        current_tickers = {p.ticker for p in portfolio.positions}

        eligible_incoming = [
            c for c in stock_candidates
            if (
                c.entry_class == EntryClass.STRONG_ENTRY
                and c.gate_result.gate4_restrength == ReStrengtheningGate.PREMIUM_PASS
                and c.ticker not in current_tickers
                and c.ticker not in _planned_tickers
                and c.ticker not in _executed
            )
        ]

        if not eligible_incoming:
            excluded_info = []
            if _planned_tickers:
                excluded_info.append(f"planned={_planned_tickers}")
            if _executed:
                excluded_info.append(f"executed={_executed}")
            logger.debug(
                f"[ReplacementEvaluator] No eligible incoming "
                f"({'excluding ' + ', '.join(excluded_info) if excluded_info else 'no candidates'})"
            )
            return None

        weakening = [p for p in portfolio.positions if self._is_weakening(p)]
        if not weakening:
            logger.debug("[ReplacementEvaluator] No weakening positions (condition 1)")
            return None

        weakest = min(weakening, key=lambda p: p.rs_score)
        best_new = max(eligible_incoming, key=lambda c: c.rs_score)

        rs_gap = best_new.rs_score - weakest.rs_score
        if rs_gap < self.superiority_threshold:
            logger.debug(
                f"[ReplacementEvaluator] RS gap {rs_gap:.2f} < "
                f"threshold {self.superiority_threshold} (condition 2)"
            )
            return None

        logger.info(
            f"[ReplacementEvaluator] All 3 conditions met — "
            f"{weakest.ticker} (RS={weakest.rs_score:.2f}, {weakest.holding_status.value}) "
            f"→ {best_new.ticker} (RS={best_new.rs_score:.2f}, "
            f"gate4={best_new.gate_result.gate4_restrength.value}, gap={rs_gap:.2f})"
        )
        return ReplacementAction(
            outgoing_ticker=weakest.ticker,
            incoming_ticker=best_new.ticker,
            incoming_entry_class=EntryClass.STRONG_ENTRY,
            reduction_ratio=REPLACEMENT_REDUCTION_RATIO,
            weakening_score=weakest.rs_score,
            superiority_rs_gap=rs_gap,
            incoming_human_review_flags=best_new.human_review_flags,
        )


class PriorityEngine:
    """Pipeline Layer 5 — Priority and Replacement Engine."""

    def __init__(self, portfolio_cfg: Dict[str, Any]) -> None:
        self.ranker = CandidateRanker()
        self.replacement_evaluator = ReplacementEvaluator(portfolio_cfg)

    def run(
        self,
        stock_candidates: List[StockCandidate],
        etf_candidates: List[ETFCandidate],
        market_filter: MarketFilterResult,
        portfolio: PortfolioState,
        counter: DailyEntryCounter,
        executed_tickers_today: Optional[Set[str]] = None,
    ) -> RankedActionList:
        """
        executed_tickers_today: tickers confirmed executed today.
          - Excluded from new_entries candidates.
          - Forwarded to ReplacementEvaluator to exclude from replacement incoming.
          - Does NOT affect next trading day.
        """
        _executed: Set[str] = executed_tickers_today or set()

        hold_candidates = [
            c for c in stock_candidates if c.entry_class == EntryClass.HOLD_CANDIDATE
        ]
        if hold_candidates:
            logger.info(f"[PriorityEngine] HOLD: {[c.ticker for c in hold_candidates]}")

        all_ranked = self.ranker.rank(
            stock_candidates, etf_candidates, market_filter.action_mode
        )

        available_slots = MAX_HOLDINGS - portfolio.num_holdings
        strong_remaining = MAX_STRONG_ENTRIES_PER_DAY - counter.strong_entries_today
        general_remaining = MAX_GENERAL_ENTRIES_PER_DAY - counter.general_entries_today
        current_tickers = {p.ticker for p in portfolio.positions}

        new_entries: List[RankedCandidate] = []
        rejected: List[Dict] = []

        for candidate in all_ranked:
            if candidate.ticker in current_tickers:
                rejected.append({
                    "ticker": candidate.ticker,
                    "reason": "Already in portfolio",
                    "rank": candidate.rank,
                })
                continue

            if candidate.ticker in _executed:
                rejected.append({
                    "ticker": candidate.ticker,
                    "reason": "Already executed today — skipped to prevent duplicate allocation",
                    "rank": candidate.rank,
                })
                continue

            if available_slots <= 0:
                rejected.append({
                    "ticker": candidate.ticker,
                    "reason": f"No slots (max {MAX_HOLDINGS})",
                    "rank": candidate.rank,
                })
                continue

            if candidate.entry_class == EntryClass.STRONG_ENTRY.value:
                if strong_remaining <= 0:
                    rejected.append({
                        "ticker": candidate.ticker,
                        "reason": f"Daily STRONG limit ({MAX_STRONG_ENTRIES_PER_DAY}/day)",
                        "rank": candidate.rank,
                    })
                    continue
                new_entries.append(candidate)
                strong_remaining -= 1
                available_slots -= 1

            elif candidate.entry_class in (
                EntryClass.GENERAL_ENTRY.value,
                ETFEntryClass.APPROVED.value,
            ):
                if general_remaining <= 0:
                    rejected.append({
                        "ticker": candidate.ticker,
                        "reason": f"Daily GENERAL/ETF limit ({MAX_GENERAL_ENTRIES_PER_DAY}/day)",
                        "rank": candidate.rank,
                    })
                    continue
                new_entries.append(candidate)
                general_remaining -= 1
                available_slots -= 1

        planned_new_entry_tickers: Set[str] = {c.ticker for c in new_entries}

        replacement = self.replacement_evaluator.evaluate(
            stock_candidates=stock_candidates,
            portfolio=portfolio,
            counter=counter,
            planned_new_entries=len(new_entries),
            planned_new_entry_tickers=planned_new_entry_tickers,
            executed_tickers_today=_executed,
        )
        replacements = [replacement] if replacement else []

        logger.info(
            f"[PriorityEngine] entries={len(new_entries)}, "
            f"replacements={len(replacements)}, "
            f"holds={len(hold_candidates)}, "
            f"rejected={len(rejected)}"
        )
        return RankedActionList(
            new_entries=new_entries,
            replacements=replacements,
            rejected_by_priority=rejected,
            hold_candidates=hold_candidates,
        )