"""
tests/test_risk_engine.py  (v1.4)

Fix 2: Removed _make_market_filter(40). Only valid ExposureCaps (100/60/20/0) used.
Fix 3: Added test for outgoing not in portfolio → replacement_blocked immediately.

ExposureCap structural constraint:
  AGGRESSIVE          = 100
  LIMITED_AGGRESSIVE  = 60
  DEFENSIVE           = 20
  STAY_OUT            = 0
  Any other value is architecturally invalid.
"""

import pytest
from datetime import date

from risk.risk_engine import LossLimitChecker, PositionSizer, RiskEngine, estimate_stop_loss
from core.models import (
    AccountState,
    ActionMode,
    EntryClass,
    HoldingStatus,
    MarketFilterResult,
    MarketState,
    Position,
    PortfolioState,
    RankedActionList,
    RankedCandidate,
    ReplacementAction,
    RiskGateResult,
)
from data.data_loader import generate_synthetic_ohlcv

TODAY = date(2025, 1, 15)

# Valid ExposureCap values: 100, 60, 20, 0 only
_VALID_CAPS = {100, 60, 20, 0}


def _make_market_filter(cap: int) -> MarketFilterResult:
    """
    Build a MarketFilterResult with a structurally valid ExposureCap.
    Raises ValueError for non-standard caps to prevent test drift.
    """
    if cap not in _VALID_CAPS:
        raise ValueError(
            f"Invalid ExposureCap: {cap}. "
            f"Must be one of {_VALID_CAPS} (AGGRESSIVE=100, LIMITED_AGGRESSIVE=60, "
            f"DEFENSIVE=20, STAY_OUT=0)."
        )
    mode_map = {
        100: ActionMode.AGGRESSIVE,
        60: ActionMode.LIMITED_AGGRESSIVE,
        20: ActionMode.DEFENSIVE,
        0: ActionMode.STAY_OUT,
    }
    state_map = {
        100: MarketState.GOOD,
        60: MarketState.NEUTRAL,
        20: MarketState.BAD,
        0: MarketState.BAD,
    }
    return MarketFilterResult(
        market_state=state_map[cap],
        action_mode=mode_map[cap],
        exposure_cap=cap,
        pipeline_halt=(cap == 0),
        halt_reason="STAY_OUT" if cap == 0 else None,
        qqq_signal="BULL",
        spy_signal="CONFIRM",
        vix_signal="LOW",
        analysis_date=TODAY,
    )


def _make_action_list(tickers: list) -> RankedActionList:
    return RankedActionList(
        new_entries=[
            RankedCandidate(
                rank=i + 1, ticker=t, source="STOCK",
                entry_class=EntryClass.GENERAL_ENTRY.value,
                rs_score=5.0, human_review_flags=None,
            )
            for i, t in enumerate(tickers)
        ],
        replacements=[],
        rejected_by_priority=[],
    )


def _make_action_list_with_replacement(
    new_tickers: list,
    outgoing: str,
    incoming: str,
) -> RankedActionList:
    new_entries = [
        RankedCandidate(
            rank=i + 1, ticker=t, source="STOCK",
            entry_class=EntryClass.GENERAL_ENTRY.value,
            rs_score=5.0, human_review_flags=None,
        )
        for i, t in enumerate(new_tickers)
    ]
    replacements = [
        ReplacementAction(
            outgoing_ticker=outgoing,
            incoming_ticker=incoming,
            incoming_entry_class=EntryClass.STRONG_ENTRY,
            reduction_ratio=0.5,
            weakening_score=-4.0,
            superiority_rs_gap=5.0,
        )
    ]
    return RankedActionList(
        new_entries=new_entries,
        replacements=replacements,
        rejected_by_priority=[],
    )


# ─── Loss Limit Checker ───────────────────────────────────────────────────────

def test_pre_check_halt_daily(risk_config, breached_daily_account):
    checker = LossLimitChecker(risk_config)
    halt, reason = checker.check(breached_daily_account)
    assert halt is True
    assert "Daily" in reason


def test_pre_check_no_halt_healthy(risk_config, healthy_account):
    checker = LossLimitChecker(risk_config)
    halt, reason = checker.check(healthy_account)
    assert halt is False
    assert reason is None


def test_pre_check_halt_weekly(risk_config):
    account = AccountState(
        total_capital=100_000.0,
        daily_pnl_pct=-0.005,
        weekly_pnl_pct=-0.025,
        monthly_pnl_pct=-0.025,
    )
    checker = LossLimitChecker(risk_config)
    halt, reason = checker.check(account)
    assert halt is True
    assert "Weekly" in reason


def test_pre_check_halt_monthly(risk_config):
    account = AccountState(
        total_capital=100_000.0,
        daily_pnl_pct=0.0,
        weekly_pnl_pct=-0.01,
        monthly_pnl_pct=-0.055,
    )
    checker = LossLimitChecker(risk_config)
    halt, reason = checker.check(account)
    assert halt is True
    assert "Monthly" in reason


# ─── Position Sizer ───────────────────────────────────────────────────────────

def test_position_sizer_valid(risk_config):
    sizer = PositionSizer(risk_config)
    shares, dollars, risk = sizer.calculate(
        total_capital=100_000.0, entry_price=100.0, stop_loss_price=95.0,
    )
    assert shares == pytest.approx(100.0, rel=1e-3)
    assert dollars == pytest.approx(10_000.0, rel=1e-3)
    assert risk == pytest.approx(500.0, rel=1e-3)


def test_position_sizer_stop_above_entry(risk_config):
    sizer = PositionSizer(risk_config)
    shares, _, _ = sizer.calculate(
        total_capital=100_000.0, entry_price=100.0, stop_loss_price=105.0,
    )
    assert shares == 0


def test_estimate_stop_below_entry():
    df = generate_synthetic_ohlcv("TEST", n_days=60, seed=1)
    stop = estimate_stop_loss(df)
    assert stop < float(df["close"].iloc[-1])


def test_no_auto_restart(risk_config, breached_daily_account):
    engine = RiskEngine(risk_config)
    h1, _ = engine.pre_check(breached_daily_account)
    h2, _ = engine.pre_check(breached_daily_account)
    assert h1 is True and h2 is True


# ─── Exposure validation ──────────────────────────────────────────────────────

def test_exposure_includes_existing_holdings(risk_config):
    """
    Existing portfolio exposure is subtracted from cap before approving new entries.
    cap=60 (LIMITED_AGGRESSIVE), existing=55% → remaining=5%.
    """
    engine = RiskEngine(risk_config)
    capital = 100_000.0
    existing = Position(
        ticker="AVGO", entry_price=500.0, current_price=500.0,
        quantity=110.0,   # 110 * 500 = $55,000 = 55% of capital
        entry_date=TODAY,
    )
    portfolio = PortfolioState(positions=[existing], total_capital=capital)
    assert abs(portfolio.current_exposure_pct - 55.0) < 1.0

    market_filter = _make_market_filter(60)   # valid: LIMITED_AGGRESSIVE
    account = AccountState(total_capital=capital)
    df = generate_synthetic_ohlcv("NVDA", n_days=60, start_price=500.0, seed=5)
    al = _make_action_list(["NVDA"])

    result = engine.final_gate(
        action_list=al, account=account,
        market_filter=market_filter, price_data={"NVDA": df},
        portfolio_state=portfolio,
    )
    total_new = sum(e.exposure_pct for e in result.approved)
    assert total_new <= 6.0, (
        f"New exposure {total_new:.1f}% exceeds remaining cap ~5% "
        f"(cap=60%, existing=55%)"
    )


def test_no_remaining_cap_blocks_all(risk_config):
    """
    Existing holdings exceed DEFENSIVE cap (20%) → all new entries blocked.
    cap=20 (DEFENSIVE), existing=22% → remaining=0 (clamped).
    """
    engine = RiskEngine(risk_config)
    capital = 100_000.0
    existing = Position(
        ticker="AAPL", entry_price=100.0, current_price=110.0,
        quantity=200.0,   # 200 * 110 = $22,000 = 22% of capital
        entry_date=TODAY,
    )
    portfolio = PortfolioState(positions=[existing], total_capital=capital)

    market_filter = _make_market_filter(20)   # valid: DEFENSIVE
    account = AccountState(total_capital=capital)
    df = generate_synthetic_ohlcv("MSFT", n_days=60, start_price=300.0, seed=6)
    al = _make_action_list(["MSFT"])

    result = engine.final_gate(
        action_list=al, account=account,
        market_filter=market_filter, price_data={"MSFT": df},
        portfolio_state=portfolio,
    )
    assert len(result.approved) == 0
    assert len(result.blocked) >= 1
    block_reasons = " ".join(b.block_reason for b in result.blocked)
    assert "No remaining exposure" in block_reasons or "ExposureCap" in block_reasons


def test_empty_portfolio_full_cap_available(risk_config):
    """Empty portfolio → no exposure subtracted, full cap available."""
    engine = RiskEngine(risk_config)
    capital = 100_000.0
    portfolio = PortfolioState(positions=[], total_capital=capital)
    assert portfolio.current_exposure_pct == 0.0

    market_filter = _make_market_filter(100)  # valid: AGGRESSIVE
    account = AccountState(total_capital=capital)
    df = generate_synthetic_ohlcv("NVDA", n_days=60, start_price=100.0, seed=7)
    al = _make_action_list(["NVDA"])

    result = engine.final_gate(
        action_list=al, account=account,
        market_filter=market_filter, price_data={"NVDA": df},
        portfolio_state=portfolio,
    )
    exposure_blocks = [b for b in result.blocked if "ExposureCap" in b.block_reason]
    assert len(exposure_blocks) == 0


# ─── Fix 2: Valid ExposureCap guard ──────────────────────────────────────────

def test_make_market_filter_rejects_invalid_cap():
    """
    Fix 2: _make_market_filter() must raise for any cap not in {100,60,20,0}.
    This guards against test drift using architecturally invalid caps.
    """
    with pytest.raises(ValueError, match="Invalid ExposureCap"):
        _make_market_filter(40)

    with pytest.raises(ValueError, match="Invalid ExposureCap"):
        _make_market_filter(50)

    with pytest.raises(ValueError, match="Invalid ExposureCap"):
        _make_market_filter(80)


# ─── Replacement tests ────────────────────────────────────────────────────────

def test_replacement_incoming_sized_and_approved(risk_config):
    """
    Fix 1 (from v1.3): Replacement incoming must be risk-sized and validated.
    Result must appear in risk_result.replacement_approved.
    """
    engine = RiskEngine(risk_config)
    capital = 100_000.0

    outgoing_pos = Position(
        ticker="OLD", entry_price=200.0, current_price=200.0,
        quantity=50.0,    # 50 * 200 = $10,000 = 10%
        entry_date=TODAY,
    )
    portfolio = PortfolioState(positions=[outgoing_pos], total_capital=capital)
    market_filter = _make_market_filter(100)
    account = AccountState(total_capital=capital)

    incoming_df = generate_synthetic_ohlcv("NEW", n_days=60, start_price=100.0, seed=10)
    action_list = _make_action_list_with_replacement([], outgoing="OLD", incoming="NEW")

    result = engine.final_gate(
        action_list=action_list, account=account,
        market_filter=market_filter, price_data={"NEW": incoming_df},
        portfolio_state=portfolio,
    )

    assert len(result.replacement_approved) == 1
    rep = result.replacement_approved[0]
    assert rep.outgoing_ticker == "OLD"
    assert rep.incoming_ticker == "NEW"
    assert rep.incoming_position_size_shares > 0
    assert rep.incoming_stop_loss_price < float(incoming_df["close"].iloc[-1])
    assert rep.freed_exposure_pct > 0


# ─── Fix 3: outgoing not in portfolio → replacement_blocked ──────────────────

def test_fix3_outgoing_not_in_portfolio_blocks_replacement(risk_config):
    """
    Fix 3: If outgoing_ticker is not found in portfolio_state,
    the replacement must be immediately blocked with a clear reason.
    Must NOT proceed with freed_exposure_pct=0.
    """
    engine = RiskEngine(risk_config)
    capital = 100_000.0

    # Portfolio does NOT contain the outgoing ticker "OLD"
    portfolio = PortfolioState(
        positions=[],    # empty — outgoing definitely not here
        total_capital=capital,
    )

    market_filter = _make_market_filter(100)
    account = AccountState(total_capital=capital)
    incoming_df = generate_synthetic_ohlcv("NEW", n_days=60, start_price=100.0, seed=11)
    action_list = _make_action_list_with_replacement([], outgoing="OLD", incoming="NEW")

    result = engine.final_gate(
        action_list=action_list, account=account,
        market_filter=market_filter, price_data={"NEW": incoming_df},
        portfolio_state=portfolio,
    )

    assert len(result.replacement_approved) == 0, (
        "Replacement must be blocked when outgoing is not in portfolio"
    )
    assert len(result.replacement_blocked) == 1
    block = result.replacement_blocked[0]
    assert "OLD" in block.block_reason, "Block reason must name the missing ticker"
    assert "not found" in block.block_reason.lower(), (
        "Block reason must explicitly state the ticker was not found"
    )


def test_fix3_outgoing_not_found_with_other_positions(risk_config):
    """
    Fix 3: Even when portfolio has positions, if outgoing is not among them,
    replacement is blocked. Must not confuse a different ticker with the target.
    """
    engine = RiskEngine(risk_config)
    capital = 100_000.0

    # Portfolio has MSFT but NOT OLD
    portfolio = PortfolioState(
        positions=[
            Position(
                ticker="MSFT", entry_price=300.0, current_price=300.0,
                quantity=10.0, entry_date=TODAY,
            )
        ],
        total_capital=capital,
    )

    market_filter = _make_market_filter(100)
    account = AccountState(total_capital=capital)
    incoming_df = generate_synthetic_ohlcv("NEW", n_days=60, start_price=100.0, seed=12)
    action_list = _make_action_list_with_replacement([], outgoing="OLD", incoming="NEW")

    result = engine.final_gate(
        action_list=action_list, account=account,
        market_filter=market_filter, price_data={"NEW": incoming_df},
        portfolio_state=portfolio,
    )

    assert len(result.replacement_approved) == 0
    assert len(result.replacement_blocked) == 1
    assert "OLD" in result.replacement_blocked[0].block_reason


def test_fix3_outgoing_found_proceeds_normally(risk_config):
    """
    Fix 3 negative: when outgoing IS in portfolio, proceed normally (not blocked).
    """
    engine = RiskEngine(risk_config)
    capital = 100_000.0

    outgoing_pos = Position(
        ticker="OLD", entry_price=100.0, current_price=100.0,
        quantity=100.0,   # 100 * 100 = $10,000 = 10%
        entry_date=TODAY,
    )
    portfolio = PortfolioState(positions=[outgoing_pos], total_capital=capital)
    market_filter = _make_market_filter(60)   # valid: LIMITED_AGGRESSIVE
    account = AccountState(total_capital=capital)

    incoming_df = generate_synthetic_ohlcv("NEW", n_days=60, start_price=50.0, seed=13)
    action_list = _make_action_list_with_replacement([], outgoing="OLD", incoming="NEW")

    result = engine.final_gate(
        action_list=action_list, account=account,
        market_filter=market_filter, price_data={"NEW": incoming_df},
        portfolio_state=portfolio,
    )

    # Should not be in replacement_blocked due to missing outgoing
    outgoing_not_found = [
        b for b in result.replacement_blocked if "not found" in b.block_reason.lower()
    ]
    assert len(outgoing_not_found) == 0, (
        "When outgoing is found, it must not be blocked with 'not found' reason"
    )


def test_replacement_blocked_no_price_data(risk_config):
    """Replacement incoming with no price data → replacement_blocked."""
    engine = RiskEngine(risk_config)
    outgoing_pos = Position(
        ticker="OLD", entry_price=200.0, current_price=200.0,
        quantity=50.0, entry_date=TODAY,
    )
    portfolio = PortfolioState(positions=[outgoing_pos], total_capital=100_000.0)
    action_list = _make_action_list_with_replacement([], outgoing="OLD", incoming="NO_DATA")

    result = engine.final_gate(
        action_list=action_list,
        account=AccountState(total_capital=100_000.0),
        market_filter=_make_market_filter(100),
        price_data={},   # no price data for incoming
        portfolio_state=portfolio,
    )

    assert len(result.replacement_approved) == 0
    assert len(result.replacement_blocked) == 1
    assert "NO_DATA" in result.replacement_blocked[0].ticker or \
           "NO_DATA" in result.replacement_blocked[0].block_reason


def test_replacement_exposure_uses_freed_cap(risk_config):
    """
    Fix 2 (rewrite with valid cap=60):
    freed_exposure from outgoing 50% reduction is added to remaining cap.

    Setup:
      cap=60 (LIMITED_AGGRESSIVE), existing exposure=50%, remaining=10%
      outgoing holds 50% → freed = 50% * 0.5 = 25%
      effective_cap = remaining(10%) + freed(25%) = 35%
      incoming at ~$50 → risk-based size ~0.5-15% → fits within 35%
    """
    engine = RiskEngine(risk_config)
    capital = 100_000.0

    outgoing_pos = Position(
        ticker="OLD", entry_price=100.0, current_price=100.0,
        quantity=500.0,   # 500 * 100 = $50,000 = 50% of capital
        entry_date=TODAY,
    )
    portfolio = PortfolioState(positions=[outgoing_pos], total_capital=capital)
    assert abs(portfolio.current_exposure_pct - 50.0) < 1.0

    market_filter = _make_market_filter(60)   # valid: LIMITED_AGGRESSIVE, cap=60%
    # remaining = 60% - 50% = 10%
    account = AccountState(total_capital=capital)

    # freed = 50% * 0.5 = 25%
    # effective_cap = 10% + 25% = 35%
    incoming_df = generate_synthetic_ohlcv("NEW", n_days=60, start_price=50.0, seed=20)
    action_list = _make_action_list_with_replacement([], outgoing="OLD", incoming="NEW")

    result = engine.final_gate(
        action_list=action_list, account=account,
        market_filter=market_filter, price_data={"NEW": incoming_df},
        portfolio_state=portfolio,
    )

    # Should be approved: effective_cap=35% >> typical incoming exposure
    assert len(result.replacement_approved) == 1
    rep = result.replacement_approved[0]
    # freed_exposure should be close to 25% (50% * 0.5)
    assert rep.freed_exposure_pct == pytest.approx(25.0, rel=0.05)
    # incoming exposure must fit within effective_cap
    assert rep.incoming_exposure_pct <= 35.0 + 0.5   # small tolerance for ATR variation


def test_replacement_and_new_entry_both_processed(risk_config):
    """Both new_entries and replacement are independently processed in one call."""
    engine = RiskEngine(risk_config)
    capital = 100_000.0

    outgoing_pos = Position(
        ticker="OLD", entry_price=200.0, current_price=200.0,
        quantity=20.0,    # $4,000 = 4%
        entry_date=TODAY,
    )
    portfolio = PortfolioState(positions=[outgoing_pos], total_capital=capital)
    market_filter = _make_market_filter(100)
    account = AccountState(total_capital=capital)

    new_df = generate_synthetic_ohlcv("AAPL", n_days=60, start_price=100.0, seed=1)
    incoming_df = generate_synthetic_ohlcv("NEW", n_days=60, start_price=150.0, seed=2)
    action_list = _make_action_list_with_replacement(
        new_tickers=["AAPL"], outgoing="OLD", incoming="NEW"
    )

    result = engine.final_gate(
        action_list=action_list, account=account,
        market_filter=market_filter,
        price_data={"AAPL": new_df, "NEW": incoming_df},
        portfolio_state=portfolio,
    )

    assert len(result.approved) == 1
    assert result.approved[0].ticker == "AAPL"
    assert len(result.replacement_approved) == 1
    assert result.replacement_approved[0].incoming_ticker == "NEW"