"""
tests/test_portfolio_manager.py  (v1.7)

Added tests:
  - test_sync_rs_scores_rank11_to_20_covered_by_weekly_pool (Scenario A)
  - test_sync_rs_scores_weekly_pool_triggers_weakening_judgment (Scenario B)
"""

import json
import pytest
import tempfile
from datetime import date, timedelta
from pathlib import Path

from portfolio.portfolio_manager import PortfolioManager
from core.models import (
    AccountState,
    DailyEntryCounter,
    EntryClass,
    ExecutedEntry,
    HoldingStatus,
    Position,
)

TODAY = date(2025, 1, 15)
TOMORROW = TODAY + timedelta(days=1)


def make_executed(
    ticker: str,
    cls: str = EntryClass.STRONG_ENTRY.value,
    shares: float = 10.0,
    price: float = 100.0,
    replacement: bool = False,
    outgoing: str = None,
) -> ExecutedEntry:
    return ExecutedEntry(
        ticker=ticker,
        entry_class=cls,
        shares_executed=shares,
        execution_price=price,
        execution_date=TODAY,
        is_replacement=replacement,
        outgoing_ticker=outgoing,
    )


# ─── Fix 1 (v1.7): weekly_pool.pool 기반 sync 커버리지 ───────────────────────

def test_sync_rs_scores_rank11_to_20_covered_by_weekly_pool():
    """
    시나리오 A:
    - 보유 종목 AVGO: 초기 rs_score=8.0
    - stock_candidates(top 10) 기반 map에는 AVGO 없음 → 갱신 안 됨 (stale)
    - weekly_pool.pool(top 15~20 포함) 기반 map에는 AVGO: -2.5 존재
    - sync_rs_scores() 후 AVGO.rs_score == -2.5

    이 테스트는 main_runner.py에서 rs_map 소스를
    stock_candidates → weekly_pool.pool 으로 바꾼 효과를 직접 검증한다.
    """
    manager = PortfolioManager()
    manager.initialize(100_000.0, TODAY)

    manager.portfolio.positions.append(
        Position(
            ticker="AVGO",
            entry_price=600.0,
            current_price=590.0,
            quantity=5.0,
            entry_date=TODAY,
            holding_status=HoldingStatus.HOLD,
            rs_score=8.0,  # 진입 시 rs_score
        )
    )

    # ── stock_candidates(top 10) 기반 map: AVGO 없음 ─────────────────────────
    # main_runner.py v1.6 이전 방식 시뮬레이션
    top10_map = {
        "NVDA": 10.0, "MSFT": 9.5, "AAPL": 9.0,
        "TSLA": 8.5, "AMZN": 8.0, "META": 7.5,
        "GOOGL": 7.0, "ADBE": 6.5, "QCOM": 6.0, "INTC": 5.5,
    }
    manager.sync_rs_scores(top10_map)
    assert manager.portfolio.positions[0].rs_score == pytest.approx(8.0), (
        "stock_candidates(top 10) 기반 map에는 AVGO 없음 → rs_score 갱신되지 않아 stale"
    )

    # ── weekly_pool.pool(top 15~20 포함) 기반 map: AVGO 포함 ─────────────────
    # main_runner.py v1.7 이후 방식 시뮬레이션
    weekly_pool_map = {
        **top10_map,
        "AVGO": -2.5,   # 오늘 15위, rs_score=-2.5
        "AMD": -1.0,
        "CSCO": 1.5,
        "TXN": 2.0,
        "SBUX": 3.0,
    }
    manager.sync_rs_scores(weekly_pool_map)
    assert manager.portfolio.positions[0].rs_score == pytest.approx(-2.5), (
        "weekly_pool.pool 기반 map에는 AVGO 포함 → rs_score -2.5로 갱신됨"
    )


def test_sync_rs_scores_weekly_pool_triggers_weakening_judgment():
    """
    시나리오 B:
    - 보유 종목 AVGO: 초기 rs_score=8.0, holding_status=HOLD (구조 이탈 없음)
    - weakening_threshold=-3.0
    - weekly_pool 기반 sync 후 rs_score=-4.0 (< -3.0 → weakening)
    - ReplacementEvaluator는 sync 전에는 replacement를 발동하지 않아야 함
    - sync 후에는 AVGO를 weakening으로 판단해 replacement를 발동해야 함

    이 테스트는 weekly_pool 기반 sync가 ReplacementEvaluator의
    weakening 판단에 실제로 반영됨을 검증한다.
    """
    from priority.priority_engine import ReplacementEvaluator
    from core.models import (
        HumanReviewFlags,
        MarketStateGate,
        PullbackStructureGate,
        RelativeStrengthGate,
        ReStrengtheningGate,
        RiskStatusGate,
        StockCandidate,
        StockGateResult,
    )

    manager = PortfolioManager()
    manager.initialize(100_000.0, TODAY)

    # 보유 종목 AVGO: 초기 rs_score=8.0, HOLD 상태
    manager.portfolio.positions.append(
        Position(
            ticker="AVGO",
            entry_price=600.0,
            current_price=600.0,
            quantity=5.0,
            entry_date=TODAY,
            holding_status=HoldingStatus.HOLD,
            rs_score=8.0,
        )
    )

    cfg = {"weakening_threshold": -3.0, "superiority_threshold": 3.0}
    ev = ReplacementEvaluator(cfg)

    gate = StockGateResult(
        gate1_market=MarketStateGate.PASS,
        gate2_rs=RelativeStrengthGate.TOP_TIER_PASS,
        gate3_pullback=PullbackStructureGate.PASS,
        gate4_restrength=ReStrengtheningGate.PREMIUM_PASS,
        gate5_risk=RiskStatusGate.PASS,
    )
    incoming = StockCandidate(
        ticker="NVDA",
        rs_score=10.0,
        pool_rank=1,
        gate_result=gate,
        entry_class=EntryClass.STRONG_ENTRY,
        human_review_flags=HumanReviewFlags(),
        analysis_date=TODAY,
    )
    counter = DailyEntryCounter(analysis_date=TODAY)

    # ── sync 전: rs_score=8.0 → NOT weakening → replacement None ─────────────
    result_before = ev.evaluate(
        [incoming], manager.portfolio, counter, planned_new_entries=0
    )
    assert result_before is None, (
        "sync 전 rs_score=8.0은 weakening_threshold=-3.0 기준 weakening 아님 "
        "→ replacement 발동하지 않아야 함"
    )

    # ── weekly_pool 기반 sync: AVGO rs_score → -4.0 (< -3.0 → weakening) ─────
    weekly_pool_map = {"AVGO": -4.0, "NVDA": 10.0}
    manager.sync_rs_scores(weekly_pool_map)
    assert manager.portfolio.positions[0].rs_score == pytest.approx(-4.0), (
        "weekly_pool 기반 sync 후 AVGO rs_score=-4.0으로 갱신되어야 함"
    )

    # ── sync 후: rs_score=-4.0 → weakening → replacement 발동 ─────────────────
    result_after = ev.evaluate(
        [incoming], manager.portfolio, counter, planned_new_entries=0
    )
    assert result_after is not None, (
        "sync 후 rs_score=-4.0은 weakening → replacement 발동해야 함"
    )
    assert result_after.outgoing_ticker == "AVGO"
    assert result_after.incoming_ticker == "NVDA"


# ─── 기존 테스트 (v1.6에서 이월) ─────────────────────────────────────────────

def test_daily_counter_resets_on_new_day():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_file = f.name
    try:
        manager = PortfolioManager(state_file=state_file)
        manager.initialize(100_000.0, TODAY)
        manager.confirm_executions([make_executed("AAA")])
        assert manager.daily_counter.strong_entries_today == 1
        manager.save_state()

        manager2 = PortfolioManager(state_file=state_file)
        manager2.initialize(100_000.0, TOMORROW)
        assert manager2.daily_counter.strong_entries_today == 0
        assert manager2.daily_counter.analysis_date == TOMORROW
    finally:
        Path(state_file).unlink(missing_ok=True)


def test_counter_restored_same_day():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_file = f.name
    try:
        manager = PortfolioManager(state_file=state_file)
        manager.initialize(100_000.0, TODAY)
        manager.confirm_executions([
            make_executed("A", EntryClass.STRONG_ENTRY.value),
            make_executed("B", EntryClass.GENERAL_ENTRY.value),
        ])
        manager.save_state()

        manager2 = PortfolioManager(state_file=state_file)
        manager2.initialize(100_000.0, TODAY)
        assert manager2.daily_counter.strong_entries_today == 1
        assert manager2.daily_counter.general_entries_today == 1
    finally:
        Path(state_file).unlink(missing_ok=True)


def test_positions_persist():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_file = f.name
    try:
        manager = PortfolioManager(state_file=state_file)
        manager.initialize(100_000.0, TODAY)
        manager.portfolio.positions.append(
            Position(
                ticker="NVDA", entry_price=500.0, current_price=530.0,
                quantity=20.0, entry_date=TODAY,
                holding_status=HoldingStatus.STRONG_HOLD, rs_score=8.5,
            )
        )
        manager.save_state()

        manager2 = PortfolioManager(state_file=state_file)
        manager2.initialize(100_000.0, TODAY)
        assert len(manager2.portfolio.positions) == 1
        p = manager2.portfolio.positions[0]
        assert p.ticker == "NVDA"
        assert p.rs_score == pytest.approx(8.5)
    finally:
        Path(state_file).unlink(missing_ok=True)


def test_confirm_addon_merges_not_duplicates():
    manager = PortfolioManager()
    manager.initialize(100_000.0, TODAY)
    manager.confirm_executions([make_executed("NVDA", shares=10.0, price=100.0)])
    assert len(manager.portfolio.positions) == 1
    manager.confirm_executions([make_executed("NVDA", shares=5.0, price=120.0)])
    assert len(manager.portfolio.positions) == 1
    pos = manager.portfolio.positions[0]
    assert pos.quantity == pytest.approx(15.0)
    expected_avg = (100.0 * 10 + 120.0 * 5) / 15.0
    assert pos.entry_price == pytest.approx(expected_avg, rel=1e-4)


def test_sync_rs_scores_updates_matching_position():
    manager = PortfolioManager()
    manager.initialize(100_000.0, TODAY)
    manager.portfolio.positions.append(
        Position(
            ticker="NVDA", entry_price=500.0, current_price=510.0,
            quantity=10.0, entry_date=TODAY,
            holding_status=HoldingStatus.HOLD, rs_score=5.0,
        )
    )
    manager.sync_rs_scores({"NVDA": -2.5, "AAPL": 3.0})
    assert manager.portfolio.positions[0].rs_score == pytest.approx(-2.5)


def test_sync_rs_scores_does_not_affect_unmatched():
    manager = PortfolioManager()
    manager.initialize(100_000.0, TODAY)
    manager.portfolio.positions.append(
        Position(
            ticker="TSLA", entry_price=200.0, current_price=195.0,
            quantity=5.0, entry_date=TODAY, rs_score=4.2,
        )
    )
    manager.sync_rs_scores({"NVDA": 6.0, "MSFT": 3.5})
    assert manager.portfolio.positions[0].rs_score == pytest.approx(4.2)


def test_executed_history_persists_and_restores_same_day():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_file = f.name
    try:
        manager = PortfolioManager(state_file=state_file)
        manager.initialize(100_000.0, TODAY)
        manager.confirm_executions([
            ExecutedEntry(
                ticker="NVDA", entry_class=EntryClass.STRONG_ENTRY.value,
                shares_executed=10.0, execution_price=500.0, execution_date=TODAY,
            )
        ])
        assert "NVDA" in manager.executed_tickers_today
        manager.save_state()

        manager2 = PortfolioManager(state_file=state_file)
        manager2.initialize(100_000.0, TODAY)
        assert "NVDA" in manager2.executed_tickers_today
    finally:
        Path(state_file).unlink(missing_ok=True)


def test_executed_history_cleared_on_next_day():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_file = f.name
    try:
        manager = PortfolioManager(state_file=state_file)
        manager.initialize(100_000.0, TODAY)
        manager.confirm_executions([
            ExecutedEntry(
                ticker="NVDA", entry_class=EntryClass.STRONG_ENTRY.value,
                shares_executed=10.0, execution_price=500.0, execution_date=TODAY,
            )
        ])
        manager.save_state()

        manager2 = PortfolioManager(state_file=state_file)
        manager2.initialize(100_000.0, TOMORROW)
        assert "NVDA" not in manager2.executed_tickers_today
        assert len(manager2.executed_tickers_today) == 0
    finally:
        Path(state_file).unlink(missing_ok=True)


def test_no_state_file_save_does_not_crash():
    manager = PortfolioManager(state_file=None)
    manager.initialize(100_000.0, TODAY)
    manager.save_state()


def test_corrupted_state_uses_defaults():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("INVALID{{{{{")
        state_file = f.name
    try:
        manager = PortfolioManager(state_file=state_file)
        manager.initialize(50_000.0, TODAY)
        assert manager.account.total_capital == 50_000.0
        assert len(manager.portfolio.positions) == 0
    finally:
        Path(state_file).unlink(missing_ok=True)