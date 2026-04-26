MARKET_CONFIG = {
    "benchmark_tickers": {
        "qqq": "QQQ",
        "spy": "SPY",
        "vix": "VIX",
    }
}

STOCK_CONFIG = {
    "main_lookback_days": 63,
    "secondary_lookback_days": 14,
    "main_weight": 0.7,
    "secondary_weight": 0.3,
    "strong_pool_size": 20,
    "eligible_pool_size": 10,
    "top_tier_threshold": 6,
    "pool_refresh_weekday": 0,
    "pullback": {
        "min_pullback_pct": 3.0,
        "max_pullback_pct": 15.0,
        "lookback_for_high": 15,
        "vol_lookback": 5,
        "premium_score_min": 1,
        "premium_vol_decline_factor": 0.85,
        "premium_tight_pullback_max_pct": 8.0,
        "premium_sma_proximity_pct": 3.0,
    },
    "breakout": {
        "lookback_for_high": 15,
        "near_high_pct": 2.0,
        "vol_expand_factor": 1.2,
        "vol_lookback": 5,
        "premium_vol_expand_factor": 1.5,
        "premium_momentum_min_pct": 2.0,
    },
    "restrengthening": {
        "signal_lookback": 5,
        "momentum_threshold": 1.5,
        "vol_lookback": 3,
        "premium_score_min": 1,
        "premium_momentum_threshold": 5.0,
        "premium_vol_expand_factor": 1.2,
        "premium_sma_reclaim_window": 10,
    },
}

RISK_CONFIG = {
    "max_daily_loss_pct": 1.0,
    "max_weekly_loss_pct": 2.0,
    "max_monthly_loss_pct": 5.0,
    "max_risk_per_trade_pct": 0.5,
    "min_position_size_dollars": 500.0,
    "warning_buffer_pct": 0.8,
    "stop_loss_atr_multiple": 1.5,
    "stop_loss_atr_period": 14,
}