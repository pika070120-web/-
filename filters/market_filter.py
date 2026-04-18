# filters/market_filter.py  (v1.2)
# Fix: NEUTRAL 고착 문제 해결 - any_weak 로직 추가

from datetime import date
import pandas as pd
from core.models import ActionMode, MarketFilterResult, MarketState


class MarketFilter:
    def __init__(self, market_cfg: dict):
        self.cfg = market_cfg

    def _trend_signal(self, df: pd.DataFrame) -> str:
        if df is None or len(df) < 50:
            return "WEAK"
        close = df["close"]
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        current = float(close.iloc[-1])
        momentum = float((close.iloc[-1] / close.iloc[-5] - 1) * 100)

        if current > sma20 > sma50 and momentum > 0:
            return "BULL"
        elif current < sma20 and current < sma50 and momentum < 0:
            return "WEAK"
        else:
            return "NEUTRAL"

    def _vix_signal(self, df: pd.DataFrame) -> str:
        if df is None or len(df) < 5:
            return "HIGH"
        current = float(df["close"].iloc[-1])
        avg5 = float(df["close"].iloc[-5:].mean())

        if current < 18:
            return "LOW"
        elif current < 25 and current <= avg5 * 1.1:
            return "ELEVATED"
        else:
            return "HIGH"

    def _determine_state(
        self,
        qqq_signal: str,
        spy_signal: str,
        vix_signal: str,
    ) -> tuple:
        """
        명세 기반 시장 상태 판정 (v1.2)

        핵심 수정:
          - any_weak 체크 추가 → BULL+WEAK가 더 이상 NEUTRAL로 빠지지 않음
          - STAY_OUT은 WEAK 존재 + VIX HIGH 조합에서만 발동 (너무 쉽게 주지 않음)
        """
        both_bull = qqq_signal == "BULL" and spy_signal == "BULL"
        both_weak = qqq_signal == "WEAK" and spy_signal == "WEAK"
        any_weak  = qqq_signal == "WEAK" or spy_signal == "WEAK"
        any_bull  = qqq_signal == "BULL" or spy_signal == "BULL"
        vix_high  = vix_signal == "HIGH"

        # ── 좋음장 ──────────────────────────────────────────────
        # 둘 다 BULL + VIX 안정
        if both_bull and not vix_high:
            return MarketState.GOOD, ActionMode.AGGRESSIVE, 100, False, None

        # ── 중립장 ──────────────────────────────────────────────
        # 케이스 1: 둘 다 BULL이지만 VIX 불안
        if both_bull and vix_high:
            return MarketState.NEUTRAL, ActionMode.LIMITED_AGGRESSIVE, 60, False, None

        # 케이스 2: 하나만 BULL, 나머지 NEUTRAL (WEAK 없음)
        if any_bull and not any_weak:
            return MarketState.NEUTRAL, ActionMode.LIMITED_AGGRESSIVE, 60, False, None

        # 케이스 3: 둘 다 NEUTRAL (크게 나쁘진 않음)
        if not any_bull and not any_weak:
            return MarketState.NEUTRAL, ActionMode.LIMITED_AGGRESSIVE, 60, False, None

        # ── 관망 ────────────────────────────────────────────────
        # WEAK 존재 + VIX HIGH → 관망 (신규 진입 금지)
        # 단, BULL+WEAK+VIX_HIGH는 아직 관망으로 보지 않음 (나쁨장으로)
        one_neutral_one_weak = (
            (qqq_signal == "NEUTRAL" and spy_signal == "WEAK") or
            (qqq_signal == "WEAK"    and spy_signal == "NEUTRAL")
        )
        if (both_weak or one_neutral_one_weak) and vix_high:
            return (
                MarketState.BAD,
                ActionMode.STAY_OUT,
                0,
                True,
                "시장 추세 약화 + VIX 급등",
            )

        # ── 나쁨장 ──────────────────────────────────────────────
        # WEAK가 하나라도 있으면 나쁨장 (VIX가 극단적이지 않을 때)
        if any_weak:
            return MarketState.BAD, ActionMode.DEFENSIVE, 20, False, None

        # ── 안전 fallback ────────────────────────────────────────
        return MarketState.NEUTRAL, ActionMode.LIMITED_AGGRESSIVE, 60, False, None

    def run(
        self,
        qqq_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        analysis_date: date,
    ) -> MarketFilterResult:
        qqq_signal = self._trend_signal(qqq_df)
        spy_signal = self._trend_signal(spy_df)
        vix_signal = self._vix_signal(vix_df)

        state, action, cap, halt, halt_reason = self._determine_state(
            qqq_signal, spy_signal, vix_signal
        )

        return MarketFilterResult(
            market_state=state,
            action_mode=action,
            exposure_cap=cap,
            pipeline_halt=halt,
            halt_reason=halt_reason,
            qqq_signal=qqq_signal,
            spy_signal=spy_signal,
            vix_signal=vix_signal,
            analysis_date=analysis_date,
        )