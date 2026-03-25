"""
core/models.py  (v1.5)

Changes from v1.4:
  - ReplacementAction: added incoming_human_review_flags (Fix 3)
  - RiskApprovedReplacement: added incoming_human_review_flags (Fix 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional


# ─── Market Enums ─────────────────────────────────────────────────────────────

class MarketState(str, Enum):
    GOOD = "GOOD"
    NEUTRAL = "NEUTRAL"
    BAD = "BAD"


class ActionMode(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"
    LIMITED_AGGRESSIVE = "LIMITED_AGGRESSIVE"
    DEFENSIVE = "DEFENSIVE"
    STAY_OUT = "STAY_OUT"


# ─── Stock Gate Enums ─────────────────────────────────────────────────────────

class MarketStateGate(str, Enum):
    PASS = "PASS"
    EXCEPTION_PASS = "EXCEPTION_PASS"
    FAIL = "FAIL"


class RelativeStrengthGate(str, Enum):
    TOP_TIER_PASS = "TOP_TIER_PASS"
    PASS = "PASS"
    FAIL = "FAIL"


class PullbackStructureGate(str, Enum):
    PREMIUM_PASS = "PREMIUM_PASS"
    PASS = "PASS"
    FAIL = "FAIL"


class ReStrengtheningGate(str, Enum):
    PREMIUM_PASS = "PREMIUM_PASS"
    PASS = "PASS"
    FAIL = "FAIL"


class RiskStatusGate(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class EntryClass(str, Enum):
    STRONG_ENTRY = "STRONG_ENTRY"
    GENERAL_ENTRY = "GENERAL_ENTRY"
    HOLD_CANDIDATE = "HOLD_CANDIDATE"
    INELIGIBLE = "INELIGIBLE"


# ─── ETF Gate Enums ───────────────────────────────────────────────────────────

class ETFMarketGate(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class ETFTrendGate(str, Enum):
    PREMIUM_PASS = "PREMIUM_PASS"
    PASS = "PASS"
    FAIL = "FAIL"


class ETFHeatGate(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class ETFRiskGate(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class ETFEntryClass(str, Enum):
    APPROVED = "APPROVED"
    INELIGIBLE = "INELIGIBLE"


# ─── Portfolio / Holding Enums ────────────────────────────────────────────────

class HoldingStatus(str, Enum):
    STRONG_HOLD = "STRONG_HOLD"
    HOLD = "HOLD"
    PARTIAL_REDUCE = "PARTIAL_REDUCE"
    FULL_EXIT = "FULL_EXIT"


# ─── Report Enums ─────────────────────────────────────────────────────────────

class DailyConclusion(str, Enum):
    STRONG_TRADING_ALLOWED = "STRONG_TRADING_ALLOWED"
    TRADING_ALLOWED = "TRADING_ALLOWED"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    NO_TRADE = "NO_TRADE"


class ReportFormat(str, Enum):
    MARKDOWN = "MARKDOWN"
    JSON = "JSON"


# ─── Human Review Structures ──────────────────────────────────────────────────

@dataclass
class HumanGateOverride:
    """
    Human visual review override for Gate 3/4 only.

    HARD CONSTRAINT: human cannot resurrect system pre-filter FAIL.
      - System FAIL → override to PASS/PREMIUM is silently rejected
      - Valid: PASS ↔ PREMIUM, or PASS/PREMIUM → FAIL
      - Invalid: FAIL → anything (resurrection forbidden)
    """
    gate3_pullback: Optional[PullbackStructureGate] = None
    gate4_restrength: Optional[ReStrengtheningGate] = None


@dataclass
class HumanReviewFlags:
    """Set by the system to flag items needing human visual inspection."""
    pullback_quality_needed: bool = False
    restrengthening_quality_needed: bool = False
    pullback_notes: str = ""
    restrengthening_notes: str = ""
    pullback_preliminary_grade: str = "PASS"
    restrength_preliminary_grade: str = "PASS"


# ─── Gate Results ──────────────────────────────────────────────────────────────

@dataclass
class StockGateResult:
    gate1_market: MarketStateGate
    gate2_rs: RelativeStrengthGate
    gate3_pullback: PullbackStructureGate
    gate4_restrength: ReStrengtheningGate
    gate5_risk: RiskStatusGate
    failure_gate: Optional[int] = None
    failure_reason: Optional[str] = None


@dataclass
class ETFGateResult:
    gate1_market: ETFMarketGate
    gate2_trend: ETFTrendGate
    gate3_heat: ETFHeatGate
    gate4_risk: ETFRiskGate
    failure_gate: Optional[int] = None
    failure_reason: Optional[str] = None


# ─── Filter / Engine Results ───────────────────────────────────────────────────

@dataclass
class MarketFilterResult:
    market_state: MarketState
    action_mode: ActionMode
    exposure_cap: int
    pipeline_halt: bool
    halt_reason: Optional[str]
    qqq_signal: str
    spy_signal: str
    vix_signal: str
    analysis_date: date


@dataclass
class StockCandidate:
    ticker: str
    rs_score: float
    pool_rank: int
    gate_result: StockGateResult
    entry_class: EntryClass
    human_review_flags: HumanReviewFlags
    analysis_date: date


@dataclass
class ETFCandidate:
    ticker: str
    trend_score: float
    gate_result: ETFGateResult
    entry_class: ETFEntryClass
    analysis_date: date


# ─── Priority Engine Output ────────────────────────────────────────────────────

@dataclass
class RankedCandidate:
    rank: int
    ticker: str
    source: str
    entry_class: str
    rs_score: float
    human_review_flags: Optional[HumanReviewFlags]


@dataclass
class ReplacementAction:
    """
    Replacement is triggered by 3 mandatory conditions (Option B):
      1. Existing position weakening
      2. New candidate RS gap >= superiority_threshold
      3. New candidate gate4 == PREMIUM_PASS

    Structural constraints:
      - Only STRONG_ENTRY incoming
      - Max 1/day
      - Outgoing reduced by 50% (stays in portfolio)
      - Incoming must NOT already be in same-day new_entries
      - Replacement only when slot available: current + planned_new + 1 ≤ MAX_HOLDINGS
    """
    outgoing_ticker: str
    incoming_ticker: str
    incoming_entry_class: EntryClass
    reduction_ratio: float = 0.5
    weakening_score: float = 0.0
    superiority_rs_gap: float = 0.0
    incoming_human_review_flags: Optional[HumanReviewFlags] = None


@dataclass
class RankedActionList:
    new_entries: List[RankedCandidate]
    replacements: List[ReplacementAction]
    rejected_by_priority: List[Dict[str, Any]]
    hold_candidates: List[StockCandidate] = field(default_factory=list)


# ─── Portfolio State ───────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    entry_price: float
    current_price: float
    quantity: float
    entry_date: date
    holding_status: HoldingStatus = HoldingStatus.HOLD
    rs_score: float = 0.0


@dataclass
class PortfolioState:
    positions: List[Position] = field(default_factory=list)
    total_capital: float = 0.0
    cash: float = 0.0

    @property
    def num_holdings(self) -> int:
        return len(self.positions)

    @property
    def current_exposure_pct(self) -> float:
        if self.total_capital <= 0:
            return 0.0
        invested = sum(p.current_price * p.quantity for p in self.positions)
        return (invested / self.total_capital) * 100


@dataclass
class AccountState:
    total_capital: float
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    monthly_pnl_pct: float = 0.0


@dataclass
class DailyEntryCounter:
    """
    Updated ONLY via PortfolioManager.confirm_executions().
    Pipeline EOD run does NOT update this counter.
    """
    strong_entries_today: int = 0
    general_entries_today: int = 0
    replacements_today: int = 0
    analysis_date: Optional[date] = None


@dataclass
class ExecutedEntry:
    """
    A trade actually executed by the human in the broker.
    Counter increments only when confirmed via confirm_executions().
    """
    ticker: str
    entry_class: str
    shares_executed: float
    execution_price: float
    execution_date: date
    is_replacement: bool = False
    outgoing_ticker: Optional[str] = None


@dataclass
class WeeklyStockPool:
    pool: List[Dict[str, Any]]
    eligible_pool: List[Dict[str, Any]]
    last_updated: date


# ─── Risk Gate Output ──────────────────────────────────────────────────────────

@dataclass
class RiskApprovedEntry:
    ticker: str
    position_size_shares: float
    position_size_dollars: float
    stop_loss_price: float
    risk_amount: float
    exposure_pct: float
    entry_class: str


@dataclass
class RiskApprovedReplacement:
    """
    Risk-approved replacement: full details for both outgoing and incoming.
    """
    outgoing_ticker: str
    incoming_ticker: str
    reduction_ratio: float
    incoming_position_size_shares: float
    incoming_position_size_dollars: float
    incoming_stop_loss_price: float
    incoming_risk_amount: float
    incoming_exposure_pct: float
    freed_exposure_pct: float
    net_exposure_change_pct: float
    incoming_human_review_flags: Optional[HumanReviewFlags] = None


@dataclass
class RiskBlockedEntry:
    ticker: str
    block_reason: str


@dataclass
class RiskGateResult:
    approved: List[RiskApprovedEntry]
    blocked: List[RiskBlockedEntry]
    replacement_approved: List[RiskApprovedReplacement] = field(default_factory=list)
    replacement_blocked: List[RiskBlockedEntry] = field(default_factory=list)
    pipeline_halt: bool = False
    halt_reason: Optional[str] = None