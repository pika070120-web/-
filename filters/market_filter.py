from datetime import date

import pandas as pd

from core.models import ActionMode, MarketFilterResult, MarketState


class MarketFilter:
    def __init__(self, market_cfg: dict):
        self.market_cfg = market_cfg

    def run(
        self,
        qqq_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        analysis_date: date,
    ) -> MarketFilterResult:
        return MarketFilterResult(
            market_state=MarketState.GOOD,
            action_mode=ActionMode.AGGRESSIVE,
            exposure_cap=100,
            pipeline_halt=False,
            halt_reason=None,
            qqq_signal="BULL",
            spy_signal="CONFIRM",
            vix_signal="LOW",
            analysis_date=analysis_date,
        )