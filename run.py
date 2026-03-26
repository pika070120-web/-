from datetime import date

from core.models import AccountState, ActionMode, MarketFilterResult, MarketState
from strategies.stock_engine import StockEngine
from tests.helpers import (
    make_bull_df,
    make_premium_pullback_df,
    make_premium_restrength_df,
)

risk_config = {
    "max_daily_loss_pct": 1.0,
    "max_weekly_loss_pct": 2.0,
    "max_monthly_loss_pct": 5.0,
    "max_risk_per_trade_pct": 0.5,
    "min_position_size_dollars": 500.0,
    "warning_buffer_pct": 0.8,
    "stop_loss_atr_multiple": 2.0,
    "stop_loss_atr_period": 14,
}

stock_config = {
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

def main():
    print("Swing System 실행 테스트")
    print("-" * 30)

    engine = StockEngine(stock_config, risk_config)

    benchmark_df = make_bull_df(n=100, start_price=380.0, seed=10)

    universe_data = {
        "NVDA": make_premium_pullback_df(),
        "META": make_premium_restrength_df(),
    }

    market_filter = MarketFilterResult(
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

    account = AccountState(
        total_capital=100_000.0,
        daily_pnl_pct=0.003,
        weekly_pnl_pct=0.008,
        monthly_pnl_pct=0.015,
    )

    weekly_pool, candidates = engine.run(
        universe_data=universe_data,
        benchmark_df=benchmark_df,
        market_filter=market_filter,
        account=account,
        analysis_date=market_filter.analysis_date,
        force_refresh=True,
    )

    print("StockEngine.run() 실행 성공")
    print(f"weekly eligible count: {len(weekly_pool.eligible_pool)}")
    print(f"candidate count: {len(candidates)}")

    for c in candidates:
         print(
                f"{c.ticker} | class={c.entry_class.name} | "
              f"rs={c.rs_score} | fail_gate={c.gate_result.failure_gate} | "
                f"reason={c.gate_result.failure_reason}"
            )

if __name__ == "__main__":
    main()