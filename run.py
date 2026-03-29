from datetime import date
import numpy as np
from core.models import AccountState, ActionMode, MarketFilterResult, MarketState
from strategies.stock_engine import StockEngine
from data.data_loader import generate_synthetic_ohlcv
def make_bull_df(n: int = 90, start_price: float = 100.0, seed: int = 42):
    return generate_synthetic_ohlcv(
        "TEST",
        n_days=n,
        start_price=start_price,
        trend=0.003,
        volatility=0.012,
        seed=seed,
    )
def make_premium_pullback_df(seed: int = 99):
    n = 90
    np.random.seed(seed)

    closes = np.zeros(n)
    closes[0] = 200.0
    for i in range(1, n - 20):
        closes[i] = closes[i - 1] * (1 + np.random.normal(0.003, 0.008))

    peak = float(closes[n - 21])

    for i in range(n - 20, n - 5):
        progress = (i - (n - 20)) / 15
        closes[i] = peak * (1 - 0.06 * progress)

    for i in range(n - 5, n):
        closes[i] = closes[n - 6] * (1 + 0.003 * (i - (n - 6)))

    closes = np.maximum(closes, 1.0)

    df = generate_synthetic_ohlcv(
        "NVDA",
        n_days=n,
        start_price=float(closes[0]),
        trend=0.0,
        volatility=0.0,
        seed=seed,
    )
    df["close"] = np.round(closes, 2)

    volumes = np.full(n, 1_200_000.0)
    for i in range(n - 20, n):
        volumes[i] = 700_000.0 + np.random.randint(0, 50_000)
    df["volume"] = volumes

    return df
def make_premium_restrength_df(seed: int = 77):
    n = 90
    np.random.seed(seed)

    closes = np.zeros(n)
    closes[0] = 150.0
    for i in range(1, n - 15):
        closes[i] = closes[i - 1] * (1 + np.random.normal(0.002, 0.010))

    bottom = float(closes[n - 16])

    for i in range(n - 15, n - 5):
        closes[i] = bottom * (1 - 0.06 * (i - (n - 15)) / 10)

    recovery_start = float(closes[n - 6])
    for i in range(n - 5, n):
        closes[i] = recovery_start * (1 + 0.008 * (i - (n - 6)))

    closes = np.maximum(closes, 1.0)

    df = generate_synthetic_ohlcv(
        "META",
        n_days=n,
        start_price=float(closes[0]),
        trend=0.0,
        volatility=0.0,
        seed=seed,
    )
    df["close"] = np.round(closes, 2)

    volumes = np.full(n, 1_000_000.0)
    for i in range(n - 15, n - 5):
        volumes[i] = 600_000.0
    for i in range(n - 5, n):
        volumes[i] = 1_600_000.0
    df["volume"] = volumes

    return df
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
            f"{c.ticker} | class={c.entry_class.name} | rs={c.rs_score} | "
            f"market={c.gate_result.gate1_market} | "
            f"rs_gate={c.gate_result.gate2_rs} | "
            f"pullback={c.gate_result.gate3_pullback} | "
            f"restrength={c.gate_result.gate4_restrength} | "
            f"risk={c.gate_result.gate5_risk} | "
            f"fail_gate={c.gate_result.failure_gate} | "
            f"reason={c.gate_result.failure_reason}"
        )
if __name__ == "__main__":
    main()