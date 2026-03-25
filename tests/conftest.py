"""
tests/conftest.py
Pytest fixtures ONLY. Helper functions live in tests/helpers.py.
"""

import pytest
import numpy as np
from datetime import date

from tests.helpers import (
    make_bull_df,
    make_bear_df,
    make_flat_df,
    make_pullback_df,
    make_premium_pullback_df,
    make_premium_restrength_df,
)


@pytest.fixture
def bull_qqq():
    return make_bull_df(n=100, start_price=380.0, seed=10)


@pytest.fixture
def bear_qqq():
    return make_bear_df(n=100, start_price=380.0, seed=11)


@pytest.fixture
def bull_spy():
    return make_bull_df(n=100, start_price=470.0, seed=20)


@pytest.fixture
def bear_spy():
    return make_bear_df(n=100, start_price=470.0, seed=21)


@pytest.fixture
def low_vix():
    df = make_bull_df(n=100, start_price=14.0, seed=30)
    df["close"] = np.clip(df["close"], 11.0, 16.0)
    return df


@pytest.fixture
def high_vix():
    df = make_bull_df(n=100, start_price=30.0, seed=31)
    df["close"] = np.clip(df["close"], 26.0, 40.0)
    return df


@pytest.fixture
def extreme_vix():
    df = make_bull_df(n=100, start_price=50.0, seed=32)
    df["close"] = np.clip(df["close"], 45.0, 60.0)
    return df


@pytest.fixture
def pullback_df():
    return make_pullback_df()


@pytest.fixture
def premium_pullback_df():
    return make_premium_pullback_df()


@pytest.fixture
def premium_restrength_df():
    return make_premium_restrength_df()


@pytest.fixture
def healthy_account():
    from core.models import AccountState
    return AccountState(
        total_capital=100_000.0,
        daily_pnl_pct=0.003,
        weekly_pnl_pct=0.008,
        monthly_pnl_pct=0.015,
    )


@pytest.fixture
def breached_daily_account():
    from core.models import AccountState
    return AccountState(
        total_capital=100_000.0,
        daily_pnl_pct=-0.012,
        weekly_pnl_pct=-0.012,
        monthly_pnl_pct=-0.012,
    )


@pytest.fixture
def good_market_filter():
    from core.models import MarketFilterResult, MarketState, ActionMode
    return MarketFilterResult(
        market_state=MarketState.GOOD,
        action_mode=ActionMode.AGGRESSIVE,
        exposure_cap=100,
        pipeline_halt=False,
        halt_reason=None,
        qqq_signal="BULL",
        spy_signal="CONFIRM",
        vix_signal="LOW",
        analysis_date=date(2025, 1, 15),
    )


@pytest.fixture
def neutral_market_filter():
    from core.models import MarketFilterResult, MarketState, ActionMode
    return MarketFilterResult(
        market_state=MarketState.NEUTRAL,
        action_mode=ActionMode.LIMITED_AGGRESSIVE,
        exposure_cap=60,
        pipeline_halt=False,
        halt_reason=None,
        qqq_signal="NEUTRAL",
        spy_signal="NEUTRAL",
        vix_signal="ELEVATED",
        analysis_date=date(2025, 1, 15),
    )


@pytest.fixture
def defensive_market_filter():
    from core.models import MarketFilterResult, MarketState, ActionMode
    return MarketFilterResult(
        market_state=MarketState.BAD,
        action_mode=ActionMode.DEFENSIVE,
        exposure_cap=20,
        pipeline_halt=False,
        halt_reason=None,
        qqq_signal="BEAR",
        spy_signal="CONFIRM",
        vix_signal="ELEVATED",
        analysis_date=date(2025, 1, 15),
    )


@pytest.fixture
def market_config():
    return {
        "qqq": {
            "short_trend_lookback": 20,
            "long_trend_lookback": 50,
            "momentum_period": 21,
            "momentum_threshold": 3.0,
        },
        "spy": {
            "short_trend_lookback": 20,
            "long_trend_lookback": 50,
        },
        "vix": {
            "low_threshold": 16.0,
            "elevated_threshold": 20.0,
            "high_threshold": 25.0,
            "extreme_threshold": 35.0,
            "spike_lookback": 5,
            "spike_threshold": 30.0,
        },
    }


@pytest.fixture
def risk_config():
    return {
        "max_daily_loss_pct": 1.0,
        "max_weekly_loss_pct": 2.0,
        "max_monthly_loss_pct": 5.0,
        "max_risk_per_trade_pct": 0.5,
        "min_position_size_dollars": 500.0,
        "warning_buffer_pct": 0.8,
        "stop_loss_atr_multiplier": 2.0,
        "stop_loss_atr_period": 14,
    }


@pytest.fixture
def stock_config():
    return {
        "main_lookback_days": 63,
        "secondary_lookback_days": 21,
        "main_weight": 0.7,
        "secondary_weight": 0.3,
        "strong_pool_size": 20,
        "eligible_pool_size": 10,
        "top_tier_threshold": 5,
        "pool_refresh_weekday": 0,
        "pullback": {
            "min_pullback_pct": 3.0,
            "max_pullback_pct": 15.0,
            "lookback_for_high": 20,
            "vol_lookback": 5,
            "premium_score_min": 2,
            "premium_vol_decline_factor": 0.85,
            "premium_tight_pullback_max_pct": 8.0,
            "premium_sma_proximity_pct": 3.0,
        },
        "restrengthening": {
            "signal_lookback": 5,
            "momentum_threshold": 1.5,
            "vol_lookback": 3,
            "premium_score_min": 2,
            "premium_momentum_threshold": 3.0,
            "premium_vol_expand_factor": 1.2,
            "premium_sma_reclaim_window": 10,
        },
    }