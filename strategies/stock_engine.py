"""
strategies/stock_engine.py  (v1.2)

Fix 1: Human override cannot resurrect a system pre-filter FAIL.
  - If gate3_prefilter returns FAIL → human override to PASS/PREMIUM is rejected with warning
  - If gate4_prefilter returns FAIL → same constraint applies
  - Human override is valid ONLY when system did NOT FAIL:
      PASS → PREMIUM_PASS   (upgrade)
      PREMIUM_PASS → PASS   (downgrade)
      PASS or PREMIUM_PASS → FAIL  (reject)
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.models import (
    AccountState,
    ActionMode,
    EntryClass,
    HumanGateOverride,
    HumanReviewFlags,
    MarketFilterResult,
    MarketState,
    MarketStateGate,
    PullbackStructureGate,
    RelativeStrengthGate,
    ReStrengtheningGate,
    RiskStatusGate,
    StockCandidate,
    StockGateResult,
    WeeklyStockPool,
)

logger = logging.getLogger(__name__)


# ─── RS Score ─────────────────────────────────────────────────────────────────

def compute_rs_score(
    ticker_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    main_lookback: int,
    secondary_lookback: int,
    main_weight: float,
    secondary_weight: float,
) -> float:
    close_t = ticker_df["close"]
    close_b = benchmark_df["close"]

    min_len = min(len(close_t), len(close_b))
    if min_len < main_lookback:
        return 0.0

    close_t = close_t.iloc[-min_len:]
    close_b = close_b.iloc[-min_len:]

    def pct(s: pd.Series, p: int) -> float:
        if len(s) < p:
            return 0.0
        base = float(s.iloc[-p])
        return 0.0 if base == 0 else float((s.iloc[-1] / base - 1) * 100)

    main_e = pct(close_t, main_lookback) - pct(close_b, main_lookback)
    sec_e = pct(close_t, secondary_lookback) - pct(close_b, secondary_lookback)
    total_w = main_weight + secondary_weight
    return round((main_e * main_weight + sec_e * secondary_weight) / total_w, 4)


# ─── Universe Screener ────────────────────────────────────────────────────────

class UniverseScreener:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.main_lookback: int = cfg["main_lookback_days"]
        self.secondary_lookback: int = cfg["secondary_lookback_days"]
        self.main_weight: float = cfg["main_weight"]
        self.secondary_weight: float = cfg["secondary_weight"]
        self.strong_pool_size: int = cfg["strong_pool_size"]
        self.eligible_pool_size: int = cfg["eligible_pool_size"]
        self.refresh_weekday: int = cfg.get("pool_refresh_weekday", 0)

    def should_refresh(self, pool: Optional[WeeklyStockPool], today: date) -> bool:
        if pool is None:
            return True
        return today.weekday() == self.refresh_weekday and pool.last_updated != today

    def build(
        self,
        universe_data: Dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame,
        analysis_date: date,
    ) -> WeeklyStockPool:
        scores: List[Dict[str, Any]] = []
        for ticker, df in universe_data.items():
            if df is None or len(df) < self.main_lookback:
                continue
            try:
                rs = compute_rs_score(
                    df, benchmark_df,
                    self.main_lookback, self.secondary_lookback,
                    self.main_weight, self.secondary_weight,
                )
                scores.append({"ticker": ticker, "rs_score": rs})
            except Exception as exc:
                logger.debug(f"RS skipped for {ticker}: {exc}")

        scores.sort(key=lambda x: x["rs_score"], reverse=True)
        pool = [
            {"ticker": s["ticker"], "rs_score": s["rs_score"], "rank": i + 1}
            for i, s in enumerate(scores[: self.strong_pool_size])
        ]
        eligible = pool[: self.eligible_pool_size]
        logger.info(
            f"[UniverseScreener] pool={len(pool)}, eligible={len(eligible)} "
            f"(top: {pool[0]['ticker'] if pool else 'N/A'})"
        )
        return WeeklyStockPool(pool=pool, eligible_pool=eligible, last_updated=analysis_date)


# ─── Quality Helpers ──────────────────────────────────────────────────────────

def _safe_vol_mean(df: pd.DataFrame, start_from_end: int, end_from_end: int) -> float:
    if "volume" not in df.columns or len(df) < end_from_end:
        return 0.0
    if start_from_end == 0:
        vol = df["volume"].iloc[-end_from_end:]
    else:
        vol = df["volume"].iloc[-end_from_end: -start_from_end]
    return float(vol.mean()) if not vol.empty else 0.0


def _sma_val(series: pd.Series, window: int) -> Optional[float]:
    if len(series) < window:
        return None
    v = float(series.rolling(window).mean().iloc[-1])
    return None if np.isnan(v) else v


# ─── Gate Evaluator ───────────────────────────────────────────────────────────

class GateEvaluator:
    """
    Evaluates all 5 gates for a single stock candidate.

    Fix 1 contract (enforced in evaluate()):
      Human override is accepted ONLY when system pre-filter result is not FAIL.
      System FAIL → override silently rejected, FAIL is kept, warning logged.
      Human can: PASS↔PREMIUM, or PASS/PREMIUM→FAIL.
      Human cannot: FAIL→anything (resurrection forbidden).
    """

    def __init__(self, stock_cfg: Dict[str, Any], risk_cfg: Dict[str, Any]) -> None:
        pb = stock_cfg["pullback"]
        rs = stock_cfg["restrengthening"]

        self.min_pullback: float = pb["min_pullback_pct"]
        self.max_pullback: float = pb["max_pullback_pct"]
        self.pullback_high_window: int = pb["lookback_for_high"]
        self.pb_vol_lookback: int = pb.get("vol_lookback", 5)
        self.pb_premium_score_min: int = pb.get("premium_score_min", 2)
        self.pb_premium_vol_decline: float = pb.get("premium_vol_decline_factor", 0.85)
        self.pb_premium_tight_max: float = pb.get("premium_tight_pullback_max_pct", 8.0)
        self.pb_premium_sma_prox: float = pb.get("premium_sma_proximity_pct", 3.0)

        self.restrength_window: int = rs["signal_lookback"]
        self.restrength_threshold: float = rs["momentum_threshold"]
        self.rs_vol_lookback: int = rs.get("vol_lookback", 3)
        self.rs_premium_score_min: int = rs.get("premium_score_min", 2)
        self.rs_premium_momentum: float = rs.get("premium_momentum_threshold", 3.0)
        self.rs_premium_vol_expand: float = rs.get("premium_vol_expand_factor", 1.2)
        self.rs_premium_sma_window: int = rs.get("premium_sma_reclaim_window", 10)

        self.top_tier_n: int = stock_cfg["top_tier_threshold"]

        bo = stock_cfg.get("breakout", {})
        self._bo_cfg = {
            "near_high_pct":            bo.get("near_high_pct", 2.0),
            "vol_expand_factor":        bo.get("vol_expand_factor", 1.2),
            "vol_lookback":             bo.get("vol_lookback", 5),
            "premium_vol_expand_factor":bo.get("premium_vol_expand_factor", 1.5),
            "premium_momentum_min_pct": bo.get("premium_momentum_min_pct", 2.0),
        }

        self.daily_limit: float = risk_cfg["max_daily_loss_pct"] / 100
        self.weekly_limit: float = risk_cfg["max_weekly_loss_pct"] / 100
        self.monthly_limit: float = risk_cfg["max_monthly_loss_pct"] / 100
        self.warning_buffer: float = risk_cfg.get("warning_buffer_pct", 0.8)

    # ── Gate 1 ────────────────────────────────────────────────────────────────

    def gate1(self, action_mode: ActionMode) -> MarketStateGate:
        if action_mode in (ActionMode.AGGRESSIVE, ActionMode.LIMITED_AGGRESSIVE):
            return MarketStateGate.PASS
        if action_mode == ActionMode.DEFENSIVE:
            return MarketStateGate.EXCEPTION_PASS
        return MarketStateGate.FAIL

    # ── Gate 2 ────────────────────────────────────────────────────────────────

    def gate2(self, pool_rank: int) -> RelativeStrengthGate:
        if pool_rank <= self.top_tier_n:
            return RelativeStrengthGate.TOP_TIER_PASS
        return RelativeStrengthGate.PASS

    # ── Gate 3 Pre-filter ─────────────────────────────────────────────────────

    def gate3_prefilter(
        self, df: pd.DataFrame
    ) -> Tuple[PullbackStructureGate, str, bool]:
        """
        System pre-filter for pullback structure.
        두 가지 경로:
          1. Pullback 경로: 3~15% 눌림 후 재강세
          2. Breakout 경로: 신고가 근처(0~2%) + 거래량 동반
        FAIL = hard numeric failure; human cannot override this upward.
        """
        close = df["close"]
        if len(close) < self.pullback_high_window + 1:
            return PullbackStructureGate.FAIL, "Insufficient data for pullback analysis", False

        recent_high = float(close.iloc[-self.pullback_high_window:].max())
        current = float(close.iloc[-1])

        if recent_high <= 0 or current <= 0:
            return PullbackStructureGate.FAIL, "Invalid price data", False

        drawdown_pct = (recent_high - current) / recent_high * 100

        # ── Breakout 경로 ──────────────────────────────────────
        if drawdown_pct < self.min_pullback:
            return self._gate3_breakout(df, drawdown_pct, recent_high, current)

        # ── Pullback 경로 ──────────────────────────────────────
        if drawdown_pct > self.max_pullback:
            return (
                PullbackStructureGate.FAIL,
                f"Pullback too deep: {drawdown_pct:.1f}% > max {self.max_pullback}%",
                False,
            )

        score, quality_notes = self._gate3_quality_score(df, drawdown_pct)
        is_premium = score >= self.pb_premium_score_min
        prelim = PullbackStructureGate.PREMIUM_PASS if is_premium else PullbackStructureGate.PASS

        notes = (
            f"[Pullback] {drawdown_pct:.1f}% from {self.pullback_high_window}-bar high "
            f"[{'PREMIUM' if is_premium else 'STANDARD'}, score={score}/3: "
            f"{', '.join(quality_notes) or 'basic range only'}]. "
            f"⚠️ Human review required."
        )
        return prelim, notes, True

    def _gate3_breakout(
        self,
        df: pd.DataFrame,
        drawdown_pct: float,
        recent_high: float,
        current: float,
    ) -> Tuple[PullbackStructureGate, str, bool]:
        """
        Breakout 경로: 신고가 근처 + 거래량 동반 확인
        """
        bo = self._bo_cfg

        # 신고가 근처 아니면 탈락
        if drawdown_pct > bo["near_high_pct"]:
            return (
                PullbackStructureGate.FAIL,
                f"Pullback too shallow for pullback entry ({drawdown_pct:.1f}%) "
                f"and not near high enough for breakout (>{bo['near_high_pct']}%)",
                False,
            )

        # 거래량 확인
        recent_vol = _safe_vol_mean(df, 0, bo["vol_lookback"])
        prior_vol  = _safe_vol_mean(df, bo["vol_lookback"], self.pullback_high_window)

        if prior_vol <= 0:
            return (
                PullbackStructureGate.FAIL,
                f"Breakout: insufficient volume data",
                False,
            )

        vol_ratio = recent_vol / prior_vol
        if vol_ratio < bo["vol_expand_factor"]:
            return (
                PullbackStructureGate.FAIL,
                f"Breakout: volume not expanding ({vol_ratio:.2f}x < {bo['vol_expand_factor']}x required)",
                False,
            )

        # Premium 판단: 거래량 더 강하고 + 단기 모멘텀 확인
        close = df["close"]
        momentum_5 = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0.0
        is_premium = (
            vol_ratio >= bo["premium_vol_expand_factor"]
            and momentum_5 >= bo["premium_momentum_min_pct"]
        )
        prelim = PullbackStructureGate.PREMIUM_PASS if is_premium else PullbackStructureGate.PASS

        notes = (
            f"[Breakout] {drawdown_pct:.1f}% from high (near new high), "
            f"vol={vol_ratio:.2f}x, momentum={momentum_5:+.1f}% "
            f"[{'PREMIUM' if is_premium else 'STANDARD'}]. "
            f"⚠️ Human review required."
        )
        return prelim, notes, True

    def _gate3_quality_score(
        self, df: pd.DataFrame, drawdown_pct: float
    ) -> Tuple[int, List[str]]:
        score = 0
        notes: List[str] = []
        close = df["close"]

        recent_vol = _safe_vol_mean(df, 0, self.pb_vol_lookback)
        prior_vol = _safe_vol_mean(df, self.pb_vol_lookback, self.pullback_high_window)
        if prior_vol > 0 and (recent_vol / prior_vol) <= self.pb_premium_vol_decline:
            score += 1
            notes.append(f"vol contraction ({recent_vol / prior_vol:.0%})")

        if drawdown_pct <= self.pb_premium_tight_max:
            score += 1
            notes.append(f"tight ({drawdown_pct:.1f}%)")

        sma20 = _sma_val(close, 20)
        if sma20 and sma20 > 0:
            prox = abs(float(close.iloc[-1]) - sma20) / sma20 * 100
            if prox <= self.pb_premium_sma_prox:
                score += 1
                notes.append(f"near 20-SMA ({prox:.1f}%)")

        return score, notes

    # ── Gate 4 Pre-filter ─────────────────────────────────────────────────────

    def gate4_prefilter(
        self, df: pd.DataFrame
    ) -> Tuple[ReStrengtheningGate, str, bool]:
        """
        System pre-filter for re-strengthening signal.
        Returns (preliminary_grade, notes, human_review_required).
        FAIL = hard numeric failure; human cannot override this upward.
        """
        close = df["close"]
        if len(close) < self.restrength_window + 1:
            return ReStrengtheningGate.FAIL, "Insufficient data for signal analysis", False

        recent = close.iloc[-self.restrength_window:]
        base = float(recent.iloc[0])
        if base <= 0:
            return ReStrengtheningGate.FAIL, "Invalid price data", False

        momentum = float((recent.iloc[-1] / base - 1) * 100)

        if momentum < self.restrength_threshold:
            return (
                ReStrengtheningGate.FAIL,
                f"No re-strengthening: momentum={momentum:.2f}% < {self.restrength_threshold}%",
                False,
            )

        score, quality_notes = self._gate4_quality_score(df, momentum)
        is_premium = score >= self.rs_premium_score_min
        prelim = ReStrengtheningGate.PREMIUM_PASS if is_premium else ReStrengtheningGate.PASS

        notes = (
            f"Re-strengthening: +{momentum:.2f}% over {self.restrength_window} bars "
            f"[{'PREMIUM' if is_premium else 'STANDARD'}, score={score}/3: "
            f"{', '.join(quality_notes) or 'basic momentum only'}]. "
            f"⚠️ Human review required."
        )
        return prelim, notes, True

    def _gate4_quality_score(
        self, df: pd.DataFrame, momentum: float
    ) -> Tuple[int, List[str]]:
        score = 0
        notes: List[str] = []
        close = df["close"]

        if momentum >= self.rs_premium_momentum:
            score += 1
            notes.append(f"strong momentum (+{momentum:.1f}%)")

        recent_vol = _safe_vol_mean(df, 0, self.rs_vol_lookback)
        pb_vol = _safe_vol_mean(df, self.rs_vol_lookback, self.restrength_window)
        if pb_vol > 0 and (recent_vol / pb_vol) >= self.rs_premium_vol_expand:
            score += 1
            notes.append(f"vol expanding ({recent_vol / pb_vol:.0%})")

        sma = _sma_val(close, self.rs_premium_sma_window)
        if sma and float(close.iloc[-1]) > sma:
            score += 1
            notes.append(f"above {self.rs_premium_sma_window}-SMA")

        return score, notes

    # ── Gate 5 ────────────────────────────────────────────────────────────────

    def gate5(self, account: AccountState) -> RiskStatusGate:
        if (
            account.daily_pnl_pct <= -self.daily_limit
            or account.weekly_pnl_pct <= -self.weekly_limit
            or account.monthly_pnl_pct <= -self.monthly_limit
        ):
            return RiskStatusGate.FAIL

        warn_d = -self.daily_limit * self.warning_buffer
        warn_w = -self.weekly_limit * self.warning_buffer
        warn_m = -self.monthly_limit * self.warning_buffer
        if (
            account.daily_pnl_pct <= warn_d
            or account.weekly_pnl_pct <= warn_w
            or account.monthly_pnl_pct <= warn_m
        ):
            return RiskStatusGate.WARNING

        return RiskStatusGate.PASS

    # ── Apply Override ────────────────────────────────────────────────────────

    @staticmethod
    def _apply_override_gate3(
        prelim: PullbackStructureGate,
        override_val: Optional[PullbackStructureGate],
        notes: str,
    ) -> Tuple[PullbackStructureGate, str]:
        if override_val is None:
            return prelim, notes

        if prelim == PullbackStructureGate.FAIL:
            logger.warning(
                f"[GateEvaluator] Gate3 override to '{override_val.value}' REJECTED: "
                f"system pre-filter returned FAIL. Resurrection not allowed."
            )
            return PullbackStructureGate.FAIL, f"[Override rejected — system FAIL] {notes}"

        updated_notes = f"[Human override: {override_val.value}] {notes}"
        return override_val, updated_notes

    @staticmethod
    def _apply_override_gate4(
        prelim: ReStrengtheningGate,
        override_val: Optional[ReStrengtheningGate],
        notes: str,
    ) -> Tuple[ReStrengtheningGate, str]:
        if override_val is None:
            return prelim, notes

        if prelim == ReStrengtheningGate.FAIL:
            logger.warning(
                f"[GateEvaluator] Gate4 override to '{override_val.value}' REJECTED: "
                f"system pre-filter returned FAIL. Resurrection not allowed."
            )
            return ReStrengtheningGate.FAIL, f"[Override rejected — system FAIL] {notes}"

        updated_notes = f"[Human override: {override_val.value}] {notes}"
        return override_val, updated_notes

    # ── Full Evaluate ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        df: pd.DataFrame,
        pool_rank: int,
        action_mode: ActionMode,
        account: AccountState,
        human_override: Optional[HumanGateOverride] = None,
    ) -> Tuple[StockGateResult, HumanReviewFlags]:
        flags = HumanReviewFlags()
        ho = human_override

        g1 = self.gate1(action_mode)
        if g1 == MarketStateGate.FAIL:
            return StockGateResult(
                gate1_market=g1,
                gate2_rs=RelativeStrengthGate.FAIL,
                gate3_pullback=PullbackStructureGate.FAIL,
                gate4_restrength=ReStrengtheningGate.FAIL,
                gate5_risk=RiskStatusGate.FAIL,
                failure_gate=1,
                failure_reason="Gate1: STAY_OUT (pipeline should have halted)",
            ), flags

        g2 = self.gate2(pool_rank)

        g3_prelim, g3_notes, _ = self.gate3_prefilter(df)
        g3_override = ho.gate3_pullback if ho is not None else None
        g3, g3_notes = self._apply_override_gate3(g3_prelim, g3_override, g3_notes)

        if g3_prelim != PullbackStructureGate.FAIL:
            flags.pullback_quality_needed = True
            flags.pullback_notes = g3_notes
            flags.pullback_preliminary_grade = g3_prelim.value

        if g3 == PullbackStructureGate.FAIL:
            return StockGateResult(
                gate1_market=g1,
                gate2_rs=g2,
                gate3_pullback=PullbackStructureGate.FAIL,
                gate4_restrength=ReStrengtheningGate.FAIL,
                gate5_risk=RiskStatusGate.FAIL,
                failure_gate=3,
                failure_reason=f"Gate3: {g3_notes}",
            ), flags

        g4_prelim, g4_notes, _ = self.gate4_prefilter(df)
        g4_override = ho.gate4_restrength if ho is not None else None
        g4, g4_notes = self._apply_override_gate4(g4_prelim, g4_override, g4_notes)

        if g4_prelim != ReStrengtheningGate.FAIL:
            flags.restrengthening_quality_needed = True
            flags.restrengthening_notes = g4_notes
            flags.restrength_preliminary_grade = g4_prelim.value

        if g4 == ReStrengtheningGate.FAIL:
            return StockGateResult(
                gate1_market=g1,
                gate2_rs=g2,
                gate3_pullback=g3,
                gate4_restrength=ReStrengtheningGate.FAIL,
                gate5_risk=RiskStatusGate.FAIL,
                failure_gate=4,
                failure_reason=f"Gate4: {g4_notes}",
            ), flags

        g5 = self.gate5(account)
        fail_g5 = g5 == RiskStatusGate.FAIL

        return StockGateResult(
            gate1_market=g1,
            gate2_rs=g2,
            gate3_pullback=g3,
            gate4_restrength=g4,
            gate5_risk=g5,
            failure_gate=5 if fail_g5 else None,
            failure_reason="Gate5: risk status FAIL" if fail_g5 else None,
        ), flags


# ─── Entry Classifier ─────────────────────────────────────────────────────────

class EntryClassifier:
    def classify(
        self,
        gate: StockGateResult,
        market_filter: MarketFilterResult,
        vix_stable: bool,
    ) -> EntryClass:
        g1, g2, g3, g4, g5 = (
            gate.gate1_market,
            gate.gate2_rs,
            gate.gate3_pullback,
            gate.gate4_restrength,
            gate.gate5_risk,
        )

        if gate.failure_gate in (3, 4):
            return EntryClass.INELIGIBLE

        if (
            g1 == MarketStateGate.PASS
            and g2 == RelativeStrengthGate.TOP_TIER_PASS
            and g3 == PullbackStructureGate.PREMIUM_PASS
            and g4 == ReStrengtheningGate.PREMIUM_PASS
            and g5 == RiskStatusGate.PASS
            and market_filter.market_state == MarketState.GOOD
            and vix_stable
        ):
            return EntryClass.STRONG_ENTRY

        if (
            g1 == MarketStateGate.PASS
            and g2 in (RelativeStrengthGate.TOP_TIER_PASS, RelativeStrengthGate.PASS)
            and g3 in (PullbackStructureGate.PREMIUM_PASS, PullbackStructureGate.PASS)
            and g4 in (ReStrengtheningGate.PREMIUM_PASS, ReStrengtheningGate.PASS)
            and g5 == RiskStatusGate.PASS
        ):
            return EntryClass.GENERAL_ENTRY

        if (
            g1 in (MarketStateGate.PASS, MarketStateGate.EXCEPTION_PASS)
            and g2 == RelativeStrengthGate.TOP_TIER_PASS
            and g4 == ReStrengtheningGate.PREMIUM_PASS
            and g5 in (RiskStatusGate.PASS, RiskStatusGate.WARNING)
            and gate.failure_gate not in (3, 4)
        ):
            return EntryClass.HOLD_CANDIDATE

        return EntryClass.INELIGIBLE


# ─── Main StockEngine ─────────────────────────────────────────────────────────

class StockEngine:
    def __init__(self, stock_cfg: Dict[str, Any], risk_cfg: Dict[str, Any]) -> None:
        self.cfg = stock_cfg
        self.screener = UniverseScreener(stock_cfg)
        self.gate_evaluator = GateEvaluator(stock_cfg, risk_cfg)
        self.classifier = EntryClassifier()
        self._weekly_pool: Optional[WeeklyStockPool] = None

    def run(
        self,
        universe_data: Dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame,
        market_filter: MarketFilterResult,
        account: AccountState,
        analysis_date: date,
        force_refresh: bool = False,
        human_overrides: Optional[Dict[str, HumanGateOverride]] = None,
    ) -> Tuple[WeeklyStockPool, List[StockCandidate]]:
        if force_refresh or self.screener.should_refresh(self._weekly_pool, analysis_date):
            self._weekly_pool = self.screener.build(universe_data, benchmark_df, analysis_date)

        assert self._weekly_pool is not None
        overrides = human_overrides or {}
        vix_stable = market_filter.vix_signal in ("LOW", "ELEVATED")
        candidates: List[StockCandidate] = []

        for entry in self._weekly_pool.eligible_pool:
            ticker: str = entry["ticker"]
            rank: int = entry["rank"]
            rs_score: float = entry["rs_score"]

            df = universe_data.get(ticker)
            if df is None or len(df) < 20:
                continue

            gate_result, human_flags = self.gate_evaluator.evaluate(
                df=df,
                pool_rank=rank,
                action_mode=market_filter.action_mode,
                account=account,
                human_override=overrides.get(ticker),
            )
            entry_class = self.classifier.classify(gate_result, market_filter, vix_stable)

            candidates.append(StockCandidate(
                ticker=ticker,
                rs_score=rs_score,
                pool_rank=rank,
                gate_result=gate_result,
                entry_class=entry_class,
                human_review_flags=human_flags,
                analysis_date=analysis_date,
            ))

        by_class = {ec: 0 for ec in EntryClass}
        for c in candidates:
            by_class[c.entry_class] += 1
        logger.info(
            f"[StockEngine] STRONG={by_class[EntryClass.STRONG_ENTRY]}, "
            f"GENERAL={by_class[EntryClass.GENERAL_ENTRY]}, "
            f"HOLD={by_class[EntryClass.HOLD_CANDIDATE]}, "
            f"INELIGIBLE={by_class[EntryClass.INELIGIBLE]}"
        )
        return self._weekly_pool, candidates