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
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        recent_5 = close.iloc[-5:]
        momentum = float((recent_5.iloc[-1] / recent_5.iloc[0] - 1) * 100)

        if current > sma20 > sma50 and momentum > 0:
            return "BULL"
        elif current < sma20 < sma50 and momentum < 0:
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

        # 시장 상태 판정
        both_bull = qqq_signal == "BULL" and spy_signal == "BULL"
        both_weak = qqq_signal == "WEAK" and spy_signal == "WEAK"
        vix_low = vix_signal == "LOW"
        vix_high = vix_signal == "HIGH"

        if both_bull and not vix_high:
            market_state = MarketState.GOOD
            action_mode = ActionMode.AGGRESSIVE
            exposure_cap = 100
            pipeline_halt = False
        elif both_weak and vix_high:
            market_state = MarketState.BAD
            action_mode = ActionMode.STAY_OUT
            exposure_cap = 0
            pipeline_halt = True
        elif both_weak:
            market_state = MarketState.BAD
            action_mode = ActionMode.DEFENSIVE
            exposure_cap = 20
            pipeline_halt = False
        else:
            market_state = MarketState.NEUTRAL
            action_mode = ActionMode.LIMITED_AGGRESSIVE
            exposure_cap = 60
            pipeline_halt = False

        return MarketFilterResult(
            market_state=market_state,
            action_mode=action_mode,
            exposure_cap=exposure_cap,
            pipeline_halt=pipeline_halt,
            halt_reason="시장 추세 약화 + VIX 급등" if pipeline_halt else None,
            qqq_signal=qqq_signal,
            spy_signal=spy_signal,
            vix_signal=vix_signal,
            analysis_date=analysis_date,
        )