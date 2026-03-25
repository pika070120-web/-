"""
tests/test_priority_engine.py  (v1.6)

Covers:
- replacement 3조건
- same-day new_entry / replacement overlap 방지
- slot guard (MAX_HOLDINGS)
- executed_tickers_today 필터
- human review flags carry-through
"""

import logging
import pytest
from datetime import date

from priority.priority_engine import CandidateRanker, PriorityEngine, ReplacementEvaluator
from core.constants import MAX_HOLDINGS
from core.models import (
    ActionMode,
    DailyEntryCounter,
    EntryClass,
    ETFEntryClass,
    ETFCandidate,
    ETFGateResult,
    ETFHeatGate,
    ETFMarketGate,
    ETFRiskGate,
    ETFTrendGate,
    HoldingStatus,
    HumanReviewFlags,
    MarketStateGate,
    PortfolioState,
    Position,
    PullbackStructureGate,
    RelativeStrengthGate,
    ReStrengtheningGate,
    RiskStatusGate,
    StockCandidate,
    StockGateResult,
)

TODAY = date(2025, 1, 15)


def _make_stock(
    ticker: str,
    rs: float,
    cls: EntryClass,
    gate4: ReStrengtheningGate = ReStrengtheningGate.PASS,
    rank: int = 1,
) -> StockCandidate:
    gate = StockGateResult(
        gate1_market=MarketStateGate.PASS,
        gate2_rs=RelativeStrengthGate.TOP_TIER_PASS,
        gate3_pullback=PullbackStructureGate.PASS,
        gate4_restrength=gate4,
        gate5_risk=RiskStatusGate.PASS,
    )
    return StockCandidate(
        ticker=ticker,
        rs_score=rs,
        pool_rank=rank,
        gate_result=gate,
        entry_class=cls,
        human_review_flags=HumanReviewFlags(),
        analysis_date=TODAY,
    )


def _make_weakening(ticker: str, rs: float = -5.0) -> Position:
    return Position(
        ticker=ticker,
        entry_price=100.0,
        current_price=90.0,
        quantity=10.0,
        entry_date=TODAY,
        holding_status=HoldingStatus.PARTIAL_REDUCE,
        rs_score=rs,
    )


# ─── Replacement core conditions ──────────────────────────────────────────────

def test_all_3_conditions_required_for_replacement():
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    # Condition 3 missing: gate4 != PREMIUM_PASS
    candidates = [
        _make_stock(
            "NEW1",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PASS,
        )
    ]
    assert ev.evaluate(candidates, portfolio, counter) is None


def test_replacement_triggers_when_all_3_conditions_met():
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    candidates = [
        _make_stock(
            "NEW1",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        )
    ]

    result = ev.evaluate(candidates, portfolio, counter)
    assert result is not None
    assert result.outgoing_ticker == "OLD"
    assert result.incoming_ticker == "NEW1"


# ─── Slot guard ───────────────────────────────────────────────────────────────

def test_fix1_slot_guard_max_holdings_zero_new_entries():
    """
    Outgoing stays at 50%, so replacement still needs one extra slot.
    3 existing + 1 incoming = 4 > 3 → blocked.
    """
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    positions = [_make_weakening(f"P{i}", rs=float(-5 + i)) for i in range(MAX_HOLDINGS)]
    portfolio = PortfolioState(positions=positions, total_capital=100_000.0)

    counter = DailyEntryCounter(analysis_date=TODAY)
    candidates = [
        _make_stock(
            "NEW",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        )
    ]

    result = ev.evaluate(candidates, portfolio, counter, planned_new_entries=0)
    assert result is None


def test_fix2_precount_documented_in_log(caplog):
    """
    planned_new_entries is a pre-risk count and intentionally conservative.
    """
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)
    candidates = [
        _make_stock(
            "STRONG1",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        )
    ]

    with caplog.at_level(logging.INFO, logger="priority.priority_engine"):
        result = ev.evaluate(
            candidates,
            portfolio,
            counter,
            planned_new_entries=2,
            planned_new_entry_tickers=set(),
        )

    assert result is None
    combined_log = " ".join(caplog.messages)
    assert "pre-risk" in combined_log.lower()


# ─── Same-day duplicate ticker guards ────────────────────────────────────────

def test_fix1_same_day_new_entry_ticker_excluded_from_replacement(good_market_filter):
    """
    A ticker already selected as new_entry must not also become replacement incoming.
    """
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    engine = PriorityEngine(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    candidates = [
        _make_stock(
            "STRONG1",
            rs=9.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
    ]

    result = engine.run(
        stock_candidates=candidates,
        etf_candidates=[],
        market_filter=good_market_filter,
        portfolio=portfolio,
        counter=counter,
    )

    new_entry_tickers = {c.ticker for c in result.new_entries}
    replacement_tickers = {r.incoming_ticker for r in result.replacements}

    assert "STRONG1" in new_entry_tickers
    assert "STRONG1" not in replacement_tickers
    assert len(result.replacements) == 0


def test_fix1_different_strong_allowed_as_replacement(good_market_filter):
    """
    If STRONG1 is new_entry, STRONG2 can still be replacement incoming.
    """
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    engine = PriorityEngine(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    candidates = [
        _make_stock(
            "STRONG1",
            rs=9.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
        _make_stock(
            "STRONG2",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
    ]

    result = engine.run(
        stock_candidates=candidates,
        etf_candidates=[],
        market_filter=good_market_filter,
        portfolio=portfolio,
        counter=counter,
    )

    new_entry_tickers = {c.ticker for c in result.new_entries}
    replacement_tickers = {r.incoming_ticker for r in result.replacements}
    overlap = new_entry_tickers & replacement_tickers

    assert len(overlap) == 0
    assert len(result.replacements) <= 1
    if result.replacements:
        assert result.replacements[0].incoming_ticker not in new_entry_tickers


def test_fix1_evaluator_excludes_planned_tickers_directly():
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    candidates = [
        _make_stock(
            "STRONG1",
            rs=9.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
        _make_stock(
            "STRONG2",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
    ]

    result = ev.evaluate(
        candidates,
        portfolio,
        counter,
        planned_new_entries=1,
        planned_new_entry_tickers={"STRONG1"},
    )

    assert result is not None
    assert result.incoming_ticker == "STRONG2"


def test_fix1_no_eligible_when_all_planned():
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    candidates = [
        _make_stock(
            "STRONG1",
            rs=9.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
        _make_stock(
            "STRONG2",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
    ]

    result = ev.evaluate(
        candidates,
        portfolio,
        counter,
        planned_new_entries=2,
        planned_new_entry_tickers={"STRONG1", "STRONG2"},
    )

    assert result is None


def test_fix1_empty_planned_tickers_behaves_as_before():
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    candidates = [
        _make_stock(
            "STRONG1",
            rs=8.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        )
    ]

    result = ev.evaluate(
        candidates,
        portfolio,
        counter,
        planned_new_entries=0,
        planned_new_entry_tickers=set(),
    )
    assert result is not None
    assert result.incoming_ticker == "STRONG1"


# ─── Human review flags carry-through ────────────────────────────────────────

def test_fix3_replacement_carries_human_review_flags():
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    flags = HumanReviewFlags(
        pullback_quality_needed=True,
        restrengthening_quality_needed=True,
        pullback_notes="Tight pullback, check volume",
        restrengthening_notes="Good momentum, verify volume",
        pullback_preliminary_grade="PREMIUM_PASS",
        restrength_preliminary_grade="PREMIUM_PASS",
    )

    gate = StockGateResult(
        gate1_market=MarketStateGate.PASS,
        gate2_rs=RelativeStrengthGate.TOP_TIER_PASS,
        gate3_pullback=PullbackStructureGate.PREMIUM_PASS,
        gate4_restrength=ReStrengtheningGate.PREMIUM_PASS,
        gate5_risk=RiskStatusGate.PASS,
    )
    candidate = StockCandidate(
        ticker="STRONG1",
        rs_score=8.0,
        pool_rank=1,
        gate_result=gate,
        entry_class=EntryClass.STRONG_ENTRY,
        human_review_flags=flags,
        analysis_date=TODAY,
    )

    result = ev.evaluate([candidate], portfolio, counter)

    assert result is not None
    assert result.incoming_human_review_flags is not None
    assert result.incoming_human_review_flags.pullback_quality_needed is True
    assert result.incoming_human_review_flags.restrengthening_quality_needed is True
    assert result.incoming_human_review_flags.pullback_preliminary_grade == "PREMIUM_PASS"


def test_fix3_no_flags_when_candidate_has_empty_flags():
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    empty_flags = HumanReviewFlags(
        pullback_quality_needed=False,
        restrengthening_quality_needed=False,
    )
    candidate = _make_stock(
        "STRONG1",
        rs=8.0,
        cls=EntryClass.STRONG_ENTRY,
        gate4=ReStrengtheningGate.PREMIUM_PASS,
    )
    candidate.human_review_flags = empty_flags

    result = ev.evaluate([candidate], portfolio, counter)

    assert result is not None
    assert result.incoming_human_review_flags is not None
    assert result.incoming_human_review_flags.pullback_quality_needed is False
    assert result.incoming_human_review_flags.restrengthening_quality_needed is False


# ─── executed_tickers_today filter ───────────────────────────────────────────

def test_executed_ticker_excluded_from_new_entries(good_market_filter):
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    engine = PriorityEngine(cfg)

    candidates = [
        _make_stock("STRONG1", rs=8.0, cls=EntryClass.STRONG_ENTRY),
        _make_stock("STRONG2", rs=7.0, cls=EntryClass.STRONG_ENTRY),
    ]
    portfolio = PortfolioState(total_capital=100_000.0)
    counter = DailyEntryCounter(analysis_date=TODAY)

    result = engine.run(
        stock_candidates=candidates,
        etf_candidates=[],
        market_filter=good_market_filter,
        portfolio=portfolio,
        counter=counter,
        executed_tickers_today={"STRONG1"},
    )

    new_tickers = {c.ticker for c in result.new_entries}
    assert "STRONG1" not in new_tickers
    assert "STRONG2" in new_tickers


def test_executed_ticker_excluded_from_replacement_incoming(good_market_filter):
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    engine = PriorityEngine(cfg)

    candidates = [
        _make_stock(
            "STRONG1",
            rs=9.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
    ]
    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    result = engine.run(
        stock_candidates=candidates,
        etf_candidates=[],
        market_filter=good_market_filter,
        portfolio=portfolio,
        counter=counter,
        executed_tickers_today={"STRONG1"},
    )

    replacement_tickers = {r.incoming_ticker for r in result.replacements}
    assert "STRONG1" not in replacement_tickers
    assert len(result.replacements) == 0


def test_executed_restriction_is_same_day_only(good_market_filter):
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    engine = PriorityEngine(cfg)

    candidates = [
        _make_stock("STRONG1", rs=8.0, cls=EntryClass.STRONG_ENTRY),
    ]
    portfolio = PortfolioState(total_capital=100_000.0)
    counter = DailyEntryCounter(analysis_date=TODAY)

    result = engine.run(
        stock_candidates=candidates,
        etf_candidates=[],
        market_filter=good_market_filter,
        portfolio=portfolio,
        counter=counter,
        executed_tickers_today=set(),
    )

    new_tickers = {c.ticker for c in result.new_entries}
    assert "STRONG1" in new_tickers


def test_executed_tickers_none_param_behaves_same_as_empty_set(good_market_filter):
    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    engine = PriorityEngine(cfg)

    candidates = [
        _make_stock("GENERAL1", rs=5.0, cls=EntryClass.GENERAL_ENTRY),
    ]
    portfolio = PortfolioState(total_capital=100_000.0)
    counter = DailyEntryCounter(analysis_date=TODAY)

    result = engine.run(
        stock_candidates=candidates,
        etf_candidates=[],
        market_filter=good_market_filter,
        portfolio=portfolio,
        counter=counter,
        executed_tickers_today=None,
    )

    assert len(result.new_entries) == 1
    assert result.new_entries[0].ticker == "GENERAL1"


# ─── Ranking / integration ────────────────────────────────────────────────────

def test_ranking_strong_before_general():
    ranker = CandidateRanker()
    stocks = [
        _make_stock("G1", 5.0, EntryClass.GENERAL_ENTRY),
        _make_stock("S1", 8.0, EntryClass.STRONG_ENTRY),
    ]
    ranked = ranker.rank(stocks, [], ActionMode.AGGRESSIVE)
    tickers = [r.ticker for r in ranked]
    assert tickers.index("S1") < tickers.index("G1")


def test_etf_outranks_general_in_neutral():
    ranker = CandidateRanker()

    stocks = [
        _make_stock("G1", 5.0, EntryClass.GENERAL_ENTRY),
        _make_stock("S1", 8.0, EntryClass.STRONG_ENTRY),
    ]

    etf_gate = ETFGateResult(
        gate1_market=ETFMarketGate.PASS,
        gate2_trend=ETFTrendGate.PASS,
        gate3_heat=ETFHeatGate.PASS,
        gate4_risk=ETFRiskGate.PASS,
    )
    etfs = [
        ETFCandidate(
            ticker="ETF1",
            trend_score=6.0,
            gate_result=etf_gate,
            entry_class=ETFEntryClass.APPROVED,
            analysis_date=TODAY,
        )
    ]

    ranked = ranker.rank(stocks, etfs, ActionMode.LIMITED_AGGRESSIVE)
    tickers = [r.ticker for r in ranked]

    assert tickers.index("S1") < tickers.index("ETF1")
    assert tickers.index("ETF1") < tickers.index("G1")


def test_hold_candidates_not_in_new_entries(good_market_filter):
    engine = PriorityEngine({"weakening_threshold": -3.0, "superiority_threshold": 3.0})
    cands = [_make_stock(f"HC{i}", 5.0, EntryClass.HOLD_CANDIDATE) for i in range(2)]
    portfolio = PortfolioState(total_capital=100_000.0)
    counter = DailyEntryCounter(analysis_date=TODAY)

    result = engine.run(cands, [], good_market_filter, portfolio, counter)

    assert len(result.new_entries) == 0
    assert len(result.hold_candidates) == 2


def test_fix1_integration_slot_plus_no_overlap(good_market_filter):
    engine = PriorityEngine({"weakening_threshold": -3.0, "superiority_threshold": 3.0})

    portfolio = PortfolioState(
        positions=[_make_weakening("OLD", rs=-5.0)],
        total_capital=100_000.0,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    candidates = [
        _make_stock("G1", rs=4.0, cls=EntryClass.GENERAL_ENTRY),
        _make_stock(
            "STRONG1",
            rs=9.0,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
        _make_stock(
            "STRONG2",
            rs=8.5,
            cls=EntryClass.STRONG_ENTRY,
            gate4=ReStrengtheningGate.PREMIUM_PASS,
        ),
    ]

    result = engine.run(
        stock_candidates=candidates,
        etf_candidates=[],
        market_filter=good_market_filter,
        portfolio=portfolio,
        counter=counter,
    )

    new_entry_tickers = {c.ticker for c in result.new_entries}
    replacement_tickers = {r.incoming_ticker for r in result.replacements}
    overlap = new_entry_tickers & replacement_tickers

    assert len(overlap) == 0