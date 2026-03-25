"""
risk/risk_engine.py  (v1.5)

Fix 3: RiskApprovedReplacement now carries incoming_human_review_flags
from ReplacementAction through to the report.
No logic changes — flags are passed through unchanged.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.models import (
    AccountState,
    MarketFilterResult,
    PortfolioState,
    RankedActionList,
    RiskApprovedEntry,
    RiskApprovedReplacement,
    RiskBlockedEntry,
    RiskGateResult,
)

logger = logging.getLogger(__name__)


def estimate_stop_loss(
    df: pd.DataFrame,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
) -> float:
    """ATR-based stop loss. Floored at recent structural low."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    if len(df) < atr_period + 1:
        return float(low.iloc[-5:].min()) if len(low) >= 5 else float(close.iloc[-1]) * 0.95

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = float(tr.rolling(atr_period).mean().iloc[-1])
    current = float(close.iloc[-1])
    stop = current - (atr * atr_multiplier)
    structural_low = float(low.iloc[-atr_period:].min())
    stop = max(stop, structural_low)
    stop = min(stop, current * 0.999)
    return round(stop, 2)


class LossLimitChecker:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.daily_limit: float = cfg["max_daily_loss_pct"] / 100
        self.weekly_limit: float = cfg["max_weekly_loss_pct"] / 100
        self.monthly_limit: float = cfg["max_monthly_loss_pct"] / 100

    def check(self, account: AccountState) -> Tuple[bool, Optional[str]]:
        if account.daily_pnl_pct <= -self.daily_limit:
            return True, (
                f"Daily loss limit breached: {account.daily_pnl_pct*100:.2f}% "
                f"(limit: -{self.daily_limit*100:.1f}%)"
            )
        if account.weekly_pnl_pct <= -self.weekly_limit:
            return True, (
                f"Weekly loss limit breached: {account.weekly_pnl_pct*100:.2f}% "
                f"(limit: -{self.weekly_limit*100:.1f}%)"
            )
        if account.monthly_pnl_pct <= -self.monthly_limit:
            return True, (
                f"Monthly loss limit breached: {account.monthly_pnl_pct*100:.2f}% "
                f"(limit: -{self.monthly_limit*100:.1f}%)"
            )
        return False, None


class PositionSizer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.max_risk_pct: float = cfg["max_risk_per_trade_pct"] / 100
        self.min_position_dollars: float = cfg.get("min_position_size_dollars", 500.0)

    def calculate(
        self,
        total_capital: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> Tuple[float, float, float]:
        if entry_price <= 0 or stop_loss_price >= entry_price:
            return 0, 0.0, 0.0
        risk_dollars = total_capital * self.max_risk_pct
        risk_per_share = entry_price - stop_loss_price
        if risk_per_share <= 0:
            return 0, 0.0, 0.0
        shares = risk_dollars / risk_per_share
        position_dollars = shares * entry_price
        if position_dollars < self.min_position_dollars:
            return 0, 0.0, 0.0
        return round(shares, 4), round(position_dollars, 2), round(risk_dollars, 2)


class RiskEngine:
    """
    Pipeline Layers 1 (pre-check) and 6 (final gate).
    Hard gate — no exceptions, no automatic recovery.
    """

    def __init__(self, risk_cfg: Dict[str, Any]) -> None:
        self.cfg = risk_cfg
        self.loss_checker = LossLimitChecker(risk_cfg)
        self.sizer = PositionSizer(risk_cfg)
        self.atr_multiplier: float = risk_cfg.get("stop_loss_atr_multiplier", 2.0)
        self.atr_period: int = risk_cfg.get("stop_loss_atr_period", 14)

    def pre_check(self, account: AccountState) -> Tuple[bool, Optional[str]]:
        halt, reason = self.loss_checker.check(account)
        if halt:
            logger.warning(f"[RiskEngine] HALT: {reason}")
        return halt, reason

    def final_gate(
        self,
        action_list: RankedActionList,
        account: AccountState,
        market_filter: MarketFilterResult,
        price_data: Dict[str, pd.DataFrame],
        portfolio_state: PortfolioState,
    ) -> RiskGateResult:
        """
        Final risk gate: sizes and validates all new_entries and replacements.

        ExposureCap structural values: 100 (AGGRESSIVE), 60 (LIMITED_AGGRESSIVE),
                                       20 (DEFENSIVE), 0 (STAY_OUT).

        new_entries:
          Standard per-trade sizing + cumulative exposure deduction.

        replacements:
          - outgoing not in portfolio → replacement_blocked immediately (Fix 3 from v1.4)
          - freed_exposure = outgoing current_price * qty * 0.5 / capital
          - effective_cap = remaining_cap + freed_exposure
          - incoming sized and validated against effective_cap
          - net_exposure_change = incoming_exposure - freed_exposure
          - incoming_human_review_flags passed through from ReplacementAction (Fix 3 v1.5)
        """
        approved: List[RiskApprovedEntry] = []
        blocked: List[RiskBlockedEntry] = []
        replacement_approved: List[RiskApprovedReplacement] = []
        replacement_blocked: List[RiskBlockedEntry] = []

        current_exposure = portfolio_state.current_exposure_pct
        remaining_cap = max(0.0, float(market_filter.exposure_cap) - current_exposure)

        logger.info(
            f"[RiskEngine] ExposureCap={market_filter.exposure_cap}%, "
            f"current_exposure={current_exposure:.1f}%, "
            f"remaining_cap={remaining_cap:.1f}%"
        )

        if remaining_cap <= 0.0 and action_list.new_entries:
            logger.warning("[RiskEngine] No remaining cap — all new entries blocked")
            for candidate in action_list.new_entries:
                blocked.append(RiskBlockedEntry(
                    ticker=candidate.ticker,
                    block_reason=(
                        f"No remaining exposure: cap={market_filter.exposure_cap}%, "
                        f"current={current_exposure:.1f}%"
                    ),
                ))
        else:
            # ── Process new entries ───────────────────────────────────────────
            for candidate in action_list.new_entries:
                ticker = candidate.ticker
                df = price_data.get(ticker)

                if df is None or len(df) < max(self.atr_period + 2, 5):
                    blocked.append(RiskBlockedEntry(
                        ticker=ticker,
                        block_reason="Insufficient price data for position sizing",
                    ))
                    continue

                entry_price = float(df["close"].iloc[-1])
                stop_loss = estimate_stop_loss(df, self.atr_multiplier, self.atr_period)
                shares, dollars, risk_amount = self.sizer.calculate(
                    total_capital=account.total_capital,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                )

                if shares <= 0:
                    blocked.append(RiskBlockedEntry(
                        ticker=ticker,
                        block_reason="Position size invalid: stop too tight or below minimum",
                    ))
                    continue

                position_exposure_pct = (dollars / account.total_capital) * 100

                if position_exposure_pct > remaining_cap:
                    blocked.append(RiskBlockedEntry(
                        ticker=ticker,
                        block_reason=(
                            f"ExposureCap exceeded: need {position_exposure_pct:.1f}%, "
                            f"remaining {remaining_cap:.1f}%"
                        ),
                    ))
                    continue

                remaining_cap -= position_exposure_pct
                approved.append(RiskApprovedEntry(
                    ticker=ticker,
                    position_size_shares=shares,
                    position_size_dollars=dollars,
                    stop_loss_price=stop_loss,
                    risk_amount=risk_amount,
                    exposure_pct=round(position_exposure_pct, 2),
                    entry_class=candidate.entry_class,
                ))
                logger.info(
                    f"[RiskEngine] APPROVED {ticker}: {shares:.2f}sh @ ~${entry_price:.2f}, "
                    f"stop=${stop_loss:.2f}, risk=${risk_amount:.0f} "
                    f"({position_exposure_pct:.1f}% exposure)"
                )

        # ── Process replacements ──────────────────────────────────────────────
        for rep in action_list.replacements:
            incoming_ticker = rep.incoming_ticker
            outgoing_ticker = rep.outgoing_ticker

            # Hard block: outgoing must exist in portfolio
            outgoing_pos = next(
                (p for p in portfolio_state.positions if p.ticker == outgoing_ticker),
                None,
            )
            if outgoing_pos is None:
                reason = (
                    f"Replacement blocked: outgoing position '{outgoing_ticker}' "
                    f"not found in current portfolio state. "
                    f"Cannot calculate freed exposure. "
                    f"Possible causes: position already closed, state file stale, "
                    f"or portfolio not refreshed before pipeline run. "
                    f"Investigate portfolio state before retrying."
                )
                logger.warning(f"[RiskEngine] {reason}")
                replacement_blocked.append(RiskBlockedEntry(
                    ticker=incoming_ticker,
                    block_reason=reason,
                ))
                continue

            # Price data check for incoming
            df = price_data.get(incoming_ticker)
            if df is None or len(df) < max(self.atr_period + 2, 5):
                replacement_blocked.append(RiskBlockedEntry(
                    ticker=incoming_ticker,
                    block_reason=(
                        f"Replacement incoming '{incoming_ticker}': "
                        f"insufficient price data"
                    ),
                ))
                continue

            # Freed exposure from outgoing 50% reduction
            freed_exposure_pct = 0.0
            if account.total_capital > 0:
                outgoing_value = outgoing_pos.current_price * outgoing_pos.quantity
                freed_exposure_pct = (outgoing_value * 0.5 / account.total_capital) * 100

            logger.info(
                f"[RiskEngine] Replacement: outgoing '{outgoing_ticker}' "
                f"(qty={outgoing_pos.quantity:.2f} @ ${outgoing_pos.current_price:.2f}) "
                f"50% reduction frees {freed_exposure_pct:.1f}%"
            )

            effective_cap = remaining_cap + freed_exposure_pct

            entry_price = float(df["close"].iloc[-1])
            stop_loss = estimate_stop_loss(df, self.atr_multiplier, self.atr_period)
            shares, dollars, risk_amount = self.sizer.calculate(
                total_capital=account.total_capital,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
            )

            if shares <= 0:
                replacement_blocked.append(RiskBlockedEntry(
                    ticker=incoming_ticker,
                    block_reason=(
                        f"Replacement incoming '{incoming_ticker}': "
                        f"position size invalid"
                    ),
                ))
                continue

            incoming_exposure_pct = (dollars / account.total_capital) * 100
            net_exposure_change = incoming_exposure_pct - freed_exposure_pct

            if incoming_exposure_pct > effective_cap:
                replacement_blocked.append(RiskBlockedEntry(
                    ticker=incoming_ticker,
                    block_reason=(
                        f"Replacement incoming '{incoming_ticker}': "
                        f"ExposureCap exceeded — need {incoming_exposure_pct:.1f}%, "
                        f"effective_cap={effective_cap:.1f}% "
                        f"(remaining={remaining_cap:.1f}% + freed={freed_exposure_pct:.1f}%)"
                    ),
                ))
                continue

            remaining_cap -= net_exposure_change
            replacement_approved.append(RiskApprovedReplacement(
                outgoing_ticker=outgoing_ticker,
                incoming_ticker=incoming_ticker,
                reduction_ratio=rep.reduction_ratio,
                incoming_position_size_shares=shares,
                incoming_position_size_dollars=dollars,
                incoming_stop_loss_price=stop_loss,
                incoming_risk_amount=risk_amount,
                incoming_exposure_pct=round(incoming_exposure_pct, 2),
                freed_exposure_pct=round(freed_exposure_pct, 2),
                net_exposure_change_pct=round(net_exposure_change, 2),
                # Fix 3: carry through human review flags from PriorityEngine
                incoming_human_review_flags=rep.incoming_human_review_flags,
            ))
            logger.info(
                f"[RiskEngine] REPLACEMENT APPROVED: "
                f"OUT {outgoing_ticker} (free {freed_exposure_pct:.1f}%) "
                f"→ IN {incoming_ticker} {shares:.2f}sh @ ~${entry_price:.2f}, "
                f"stop=${stop_loss:.2f}, risk=${risk_amount:.0f}, "
                f"net={net_exposure_change:+.1f}%"
            )

        logger.info(
            f"[RiskEngine] Final gate: "
            f"{len(approved)} new approved, {len(blocked)} new blocked, "
            f"{len(replacement_approved)} replacement approved, "
            f"{len(replacement_blocked)} replacement blocked"
        )
        return RiskGateResult(
            approved=approved,
            blocked=blocked,
            replacement_approved=replacement_approved,
            replacement_blocked=replacement_blocked,
        )