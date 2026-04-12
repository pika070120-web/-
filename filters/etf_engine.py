from datetime import date
import pandas as pd
from core.models import (
    ActionMode, MarketFilterResult, MarketState,
    ETFCandidate, ETFEntryClass, ETFGateResult,
    ETFMarketGate, ETFTrendGate, ETFHeatGate, ETFRiskGate,
    AccountState,
)


class ETFEngine:
    ETF_UNIVERSE = ["QQQ", "SPY"]

    def __init__(self, market_cfg: dict, risk_cfg: dict):
        self.market_cfg = market_cfg
        self.risk_cfg = risk_cfg

    def _gate1_market(self, market_filter: MarketFilterResult) -> ETFMarketGate:
        if market_filter.action_mode in (ActionMode.AGGRESSIVE, ActionMode.LIMITED_AGGRESSIVE):
            return ETFMarketGate.PASS
        return ETFMarketGate.FAIL

    def _gate2_trend(self, df: pd.DataFrame) -> ETFTrendGate:
        if df is None or len(df) < 50:
            return ETFTrendGate.FAIL
        close = df["close"]
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        current = float(close.iloc[-1])
        momentum = float((close.iloc[-1] / close.iloc[-10] - 1) * 100)

        if current > sma20 > sma50 and momentum > 2.0:
            return ETFTrendGate.PREMIUM_PASS
        elif current > sma20 and momentum > 0:
            return ETFTrendGate.PASS
        return ETFTrendGate.FAIL

    def _gate3_heat(self, df: pd.DataFrame) -> ETFHeatGate:
        if df is None or len(df) < 20:
            return ETFHeatGate.FAIL
        close = df["close"]
        sma20 = float(close.rolling(20).mean().iloc[-1])
        current = float(close.iloc[-1])
        gap_pct = (current - sma20) / sma20 * 100

        if gap_pct > 8.0:
            return ETFHeatGate.FAIL
        return ETFHeatGate.PASS

    def _gate4_risk(self, account: AccountState) -> ETFRiskGate:
        daily_limit = self.risk_cfg["max_daily_loss_pct"] / 100
        weekly_limit = self.risk_cfg["max_weekly_loss_pct"] / 100
        monthly_limit = self.risk_cfg["max_monthly_loss_pct"] / 100
        buffer = self.risk_cfg.get("warning_buffer_pct", 0.8)

        if (
            account.daily_pnl_pct <= -daily_limit
            or account.weekly_pnl_pct <= -weekly_limit
            or account.monthly_pnl_pct <= -monthly_limit
        ):
            return ETFRiskGate.FAIL

        if (
            account.daily_pnl_pct <= -daily_limit * buffer
            or account.weekly_pnl_pct <= -weekly_limit * buffer
            or account.monthly_pnl_pct <= -monthly_limit * buffer
        ):
            return ETFRiskGate.WARNING

        return ETFRiskGate.PASS

    def run(
        self,
        universe_data: dict,
        market_filter: MarketFilterResult,
        account: AccountState,
        analysis_date: date,
    ) -> list[ETFCandidate]:
        candidates = []

        for ticker in self.ETF_UNIVERSE:
            df = universe_data.get(ticker)

            g1 = self._gate1_market(market_filter)
            if g1 == ETFMarketGate.FAIL:
                gate = ETFGateResult(
                    gate1_market=g1,
                    gate2_trend=ETFTrendGate.FAIL,
                    gate3_heat=ETFHeatGate.FAIL,
                    gate4_risk=ETFRiskGate.FAIL,
                    failure_gate=1,
                    failure_reason="Gate1: 시장 나쁨/관망",
                )
                candidates.append(ETFCandidate(
                    ticker=ticker,
                    trend_score=0.0,
                    gate_result=gate,
                    entry_class=ETFEntryClass.INELIGIBLE,
                    analysis_date=analysis_date,
                ))
                continue

            g2 = self._gate2_trend(df)
            if g2 == ETFTrendGate.FAIL:
                gate = ETFGateResult(
                    gate1_market=g1,
                    gate2_trend=g2,
                    gate3_heat=ETFHeatGate.FAIL,
                    gate4_risk=ETFRiskGate.FAIL,
                    failure_gate=2,
                    failure_reason="Gate2: ETF 추세 약함",
                )
                candidates.append(ETFCandidate(
                    ticker=ticker,
                    trend_score=0.0,
                    gate_result=gate,
                    entry_class=ETFEntryClass.INELIGIBLE,
                    analysis_date=analysis_date,
                ))
                continue

            g3 = self._gate3_heat(df)
            if g3 == ETFHeatGate.FAIL:
                gate = ETFGateResult(
                    gate1_market=g1,
                    gate2_trend=g2,
                    gate3_heat=g3,
                    gate4_risk=ETFRiskGate.FAIL,
                    failure_gate=3,
                    failure_reason="Gate3: ETF 과열",
                )
                candidates.append(ETFCandidate(
                    ticker=ticker,
                    trend_score=0.0,
                    gate_result=gate,
                    entry_class=ETFEntryClass.INELIGIBLE,
                    analysis_date=analysis_date,
                ))
                continue

            g4 = self._gate4_risk(account)
            if g4 == ETFRiskGate.FAIL:
                gate = ETFGateResult(
                    gate1_market=g1,
                    gate2_trend=g2,
                    gate3_heat=g3,
                    gate4_risk=g4,
                    failure_gate=4,
                    failure_reason="Gate4: 리스크 한도 초과",
                )
                candidates.append(ETFCandidate(
                    ticker=ticker,
                    trend_score=0.0,
                    gate_result=gate,
                    entry_class=ETFEntryClass.INELIGIBLE,
                    analysis_date=analysis_date,
                ))
                continue

            gate = ETFGateResult(
                gate1_market=g1,
                gate2_trend=g2,
                gate3_heat=g3,
                gate4_risk=g4,
            )
            candidates.append(ETFCandidate(
                ticker=ticker,
                trend_score=1.0 if g2 == ETFTrendGate.PREMIUM_PASS else 0.5,
                gate_result=gate,
                entry_class=ETFEntryClass.APPROVED,
                analysis_date=analysis_date,
            ))

        return candidates