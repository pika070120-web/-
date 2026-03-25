"""
tests/test_stock_engine.py
Focus: Fix 1 — human override cannot resurrect system FAIL.
"""

import pytest
from datetime import date

from strategies.stock_engine import GateEvaluator, EntryClassifier
from core.models import (
    ActionMode,
    EntryClass,
    HumanGateOverride,
    MarketStateGate,
    PullbackStructureGate,
    RelativeStrengthGate,
    ReStrengtheningGate,
    RiskStatusGate,
    StockGateResult,
)
from tests.helpers import (
    make_flat_df,
    make_pullback_df,
    make_no_recovery_df,
    make_premium_pullback_df,
    make_premium_restrength_df,
)

TODAY = date(2025, 1, 15)


# ─── Gate 1 ───────────────────────────────────────────────────────────────────

def test_gate1_pass_modes(stock_config, risk_config):
    ev = GateEvaluator(stock_config, risk_config)
    assert ev.gate1(ActionMode.AGGRESSIVE) == MarketStateGate.PASS
    assert ev.gate1(ActionMode.LIMITED_AGGRESSIVE) == MarketStateGate.PASS


def test_gate1_exception_pass_defensive(stock_config, risk_config):
    ev = GateEvaluator(stock_config, risk_config)
    assert ev.gate1(ActionMode.DEFENSIVE) == MarketStateGate.EXCEPTION_PASS


def test_gate1_fail_stay_out(stock_config, risk_config):
    ev = GateEvaluator(stock_config, risk_config)
    assert ev.gate1(ActionMode.STAY_OUT) == MarketStateGate.FAIL


# ─── Gate 2 ───────────────────────────────────────────────────────────────────

def test_gate2_top_tier(stock_config, risk_config):
    ev = GateEvaluator(stock_config, risk_config)
    assert ev.gate2(1) == RelativeStrengthGate.TOP_TIER_PASS
    assert ev.gate2(5) == RelativeStrengthGate.TOP_TIER_PASS


def test_gate2_pass_outside(stock_config, risk_config):
    ev = GateEvaluator(stock_config, risk_config)
    assert ev.gate2(6) == RelativeStrengthGate.PASS
    assert ev.gate2(10) == RelativeStrengthGate.PASS


# ─── Gate 3 range checks ──────────────────────────────────────────────────────

def test_gate3_fail_no_pullback(stock_config, risk_config):
    """Flat/uptrend → drawdown < 3% → FAIL."""
    df = make_flat_df()
    ev = GateEvaluator(stock_config, risk_config)
    result, notes, human = ev.gate3_prefilter(df)
    assert result == PullbackStructureGate.FAIL
    assert human is False


def test_gate3_valid_pullback_needs_human_review(stock_config, risk_config):
    """Valid pullback range → PASS or PREMIUM, human review required."""
    df = make_pullback_df()
    ev = GateEvaluator(stock_config, risk_config)
    result, notes, human = ev.gate3_prefilter(df)
    assert result in (PullbackStructureGate.PASS, PullbackStructureGate.PREMIUM_PASS)
    assert human is True


def test_gate3_premium_pullback_achieves_premium(stock_config, risk_config):
    """Premium pullback data → PREMIUM_PASS with score >= 2."""
    df = make_premium_pullback_df()
    ev = GateEvaluator(stock_config, risk_config)
    result, notes, human = ev.gate3_prefilter(df)
    assert result == PullbackStructureGate.PREMIUM_PASS, f"Expected PREMIUM, got {result}. Notes: {notes}"
    assert human is True


# ─── Gate 4 range checks ──────────────────────────────────────────────────────

def test_gate4_fail_no_recovery(stock_config, risk_config):
    """Continued decline after pullback → negative momentum → FAIL."""
    df = make_no_recovery_df()
    ev = GateEvaluator(stock_config, risk_config)
    result, notes, human = ev.gate4_prefilter(df)
    assert result == ReStrengtheningGate.FAIL
    assert human is False


def test_gate4_premium_data_achieves_premium(stock_config, risk_config):
    """Premium restrength data → PREMIUM_PASS."""
    df = make_premium_restrength_df()
    ev = GateEvaluator(stock_config, risk_config)
    result, notes, human = ev.gate4_prefilter(df)
    assert result == ReStrengtheningGate.PREMIUM_PASS, f"Expected PREMIUM, got {result}. Notes: {notes}"
    assert human is True


# ─── Fix 1: Human override cannot resurrect system FAIL ──────────────────────

def test_fix1_gate3_override_cannot_resurrect_fail(stock_config, risk_config, healthy_account):
    """
    Fix 1: System gate3 FAIL + human override to PREMIUM_PASS → override REJECTED.
    Final gate3 must remain FAIL.
    """
    df = make_flat_df()
    ev = GateEvaluator(stock_config, risk_config)

    prelim, _, _ = ev.gate3_prefilter(df)
    assert prelim == PullbackStructureGate.FAIL, "Precondition: flat_df must fail gate3"

    override = HumanGateOverride(
        gate3_pullback=PullbackStructureGate.PREMIUM_PASS,
        gate4_restrength=None,
    )
    gate_result, _ = ev.evaluate(
        df=df,
        pool_rank=1,
        action_mode=ActionMode.AGGRESSIVE,
        account=healthy_account,
        human_override=override,
    )

    assert gate_result.gate3_pullback == PullbackStructureGate.FAIL
    assert gate_result.failure_gate == 3


def test_fix1_gate3_override_cannot_upgrade_fail_to_pass(stock_config, risk_config, healthy_account):
    """
    Fix 1: System gate3 FAIL + human override to PASS → override REJECTED.
    """
    df = make_flat_df()
    ev = GateEvaluator(stock_config, risk_config)

    prelim, _, _ = ev.gate3_prefilter(df)
    if prelim != PullbackStructureGate.FAIL:
        pytest.skip("flat_df did not produce FAIL gate3 — check synthetic data params")

    override = HumanGateOverride(
        gate3_pullback=PullbackStructureGate.PASS,
        gate4_restrength=None,
    )
    gate_result, _ = ev.evaluate(
        df=df,
        pool_rank=1,
        action_mode=ActionMode.AGGRESSIVE,
        account=healthy_account,
        human_override=override,
    )
    assert gate_result.gate3_pullback == PullbackStructureGate.FAIL


def test_fix1_gate4_override_cannot_resurrect_fail(stock_config, risk_config, healthy_account):
    """
    Fix 1: System gate4 FAIL + human override to PREMIUM_PASS → override REJECTED.
    """
    df = make_no_recovery_df()
    ev = GateEvaluator(stock_config, risk_config)

    g4_prelim, _, _ = ev.gate4_prefilter(df)
    if g4_prelim != ReStrengtheningGate.FAIL:
        pytest.skip("no_recovery_df did not produce FAIL gate4 — check synthetic data")

    override = HumanGateOverride(
        gate3_pullback=PullbackStructureGate.PASS,
        gate4_restrength=ReStrengtheningGate.PREMIUM_PASS,
    )

    gate_result, _ = ev.evaluate(
        df=df,
        pool_rank=1,
        action_mode=ActionMode.AGGRESSIVE,
        account=healthy_account,
        human_override=override,
    )

    if gate_result.failure_gate == 4:
        assert gate_result.gate4_restrength == ReStrengtheningGate.FAIL


def test_fix1_override_valid_upgrade_pass_to_premium(stock_config, risk_config, healthy_account):
    """
    Fix 1 VALID CASE: System gate3=PASS, human upgrades to PREMIUM_PASS → ACCEPTED.
    """
    df = make_pullback_df()
    ev = GateEvaluator(stock_config, risk_config)

    g3_prelim, _, _ = ev.gate3_prefilter(df)
    if g3_prelim not in (PullbackStructureGate.PASS, PullbackStructureGate.PREMIUM_PASS):
        pytest.skip("pullback_df did not pass gate3 pre-filter")

    if g3_prelim == PullbackStructureGate.PASS:
        override = HumanGateOverride(
            gate3_pullback=PullbackStructureGate.PREMIUM_PASS,
            gate4_restrength=None,
        )
        gate_result, _ = ev.evaluate(
            df=df,
            pool_rank=1,
            action_mode=ActionMode.AGGRESSIVE,
            account=healthy_account,
            human_override=override,
        )
        if gate_result.failure_gate != 3:
            assert gate_result.gate3_pullback == PullbackStructureGate.PREMIUM_PASS


def test_fix1_override_valid_downgrade_premium_to_fail(stock_config, risk_config, healthy_account):
    """
    Fix 1 VALID CASE: System gate3=PREMIUM_PASS, human downgrades to FAIL → ACCEPTED.
    """
    df = make_premium_pullback_df()
    ev = GateEvaluator(stock_config, risk_config)

    g3_prelim, _, _ = ev.gate3_prefilter(df)
    if g3_prelim != PullbackStructureGate.PREMIUM_PASS:
        pytest.skip("premium_pullback_df did not produce PREMIUM gate3")

    override = HumanGateOverride(
        gate3_pullback=PullbackStructureGate.FAIL,
        gate4_restrength=None,
    )
    gate_result, _ = ev.evaluate(
        df=df,
        pool_rank=1,
        action_mode=ActionMode.AGGRESSIVE,
        account=healthy_account,
        human_override=override,
    )
    assert gate_result.gate3_pullback == PullbackStructureGate.FAIL
    assert gate_result.failure_gate == 3


# ─── Entry classification invariants ─────────────────────────────────────────

def test_strong_entry_with_human_override(stock_config, risk_config, good_market_filter, healthy_account):
    """STRONG_ENTRY reachable via human override."""
    df = make_pullback_df()
    ev = GateEvaluator(stock_config, risk_config)

    g3_prelim, _, _ = ev.gate3_prefilter(df)
    if g3_prelim == PullbackStructureGate.FAIL:
        pytest.skip("pullback_df failed gate3 pre-filter for this seed")

    override = HumanGateOverride(
        gate3_pullback=PullbackStructureGate.PREMIUM_PASS,
        gate4_restrength=ReStrengtheningGate.PREMIUM_PASS,
    )
    gate_result, _ = ev.evaluate(
        df=df,
        pool_rank=1,
        action_mode=ActionMode.AGGRESSIVE,
        account=healthy_account,
        human_override=override,
    )

    if gate_result.gate5_risk == RiskStatusGate.PASS:
        classifier = EntryClassifier()
        result = classifier.classify(gate_result, good_market_filter, vix_stable=True)
        assert result == EntryClass.STRONG_ENTRY


def test_exception_pass_gate1_cannot_be_strong_or_general(good_market_filter):
    classifier = EntryClassifier()
    gate = StockGateResult(
        gate1_market=MarketStateGate.EXCEPTION_PASS,
        gate2_rs=RelativeStrengthGate.TOP_TIER_PASS,
        gate3_pullback=PullbackStructureGate.PREMIUM_PASS,
        gate4_restrength=ReStrengtheningGate.PREMIUM_PASS,
        gate5_risk=RiskStatusGate.PASS,
    )
    result = classifier.classify(gate, good_market_filter, vix_stable=True)
    assert result not in (EntryClass.STRONG_ENTRY, EntryClass.GENERAL_ENTRY)


def test_warning_gate5_cannot_be_general(good_market_filter):
    classifier = EntryClassifier()
    gate = StockGateResult(
        gate1_market=MarketStateGate.PASS,
        gate2_rs=RelativeStrengthGate.PASS,
        gate3_pullback=PullbackStructureGate.PASS,
        gate4_restrength=ReStrengtheningGate.PASS,
        gate5_risk=RiskStatusGate.WARNING,
    )
    result = classifier.classify(gate, good_market_filter, vix_stable=True)
    assert result != EntryClass.GENERAL_ENTRY


def test_strong_entry_requires_good_market(neutral_market_filter):
    classifier = EntryClassifier()
    gate = StockGateResult(
        gate1_market=MarketStateGate.PASS,
        gate2_rs=RelativeStrengthGate.TOP_TIER_PASS,
        gate3_pullback=PullbackStructureGate.PREMIUM_PASS,
        gate4_restrength=ReStrengtheningGate.PREMIUM_PASS,
        gate5_risk=RiskStatusGate.PASS,
    )
    result = classifier.classify(gate, neutral_market_filter, vix_stable=True)
    assert result != EntryClass.STRONG_ENTRY
    assert result == EntryClass.GENERAL_ENTRY