from datetime import date, timedelta

from core.config_loader import MARKET_CONFIG, RISK_CONFIG, STOCK_CONFIG
from core.models import AccountState
from data.runtime_yf_loader import RuntimeYFinanceLoader
from filters.market_filter import MarketFilter
from filters.etf_engine import ETFEngine
from strategies.stock_engine import StockEngine
from portfolio.portfolio_manager import PortfolioManager
from reports.report_generator import ReportGenerator

PORTFOLIO_CFG = {
    "structure_break_pct": -8.0,
    "trend_weak_pct": -4.0,
    "strong_hold_rs_threshold": 5.0,
}

# 백테스트 확정 파라미터
EXECUTION_CFG = {
    "profit_target_pct": 6.5,      # 수익 목표
    "max_hold_days": 20,           # 최대 보유일
    "atr_stop_multiple": 1.5,      # ATR 손절 배수
    "cooldown_loss_pct": -10.0,    # 쿨다운 트리거 손실
    "cooldown_days": 5,            # 쿨다운 기간
    "max_monthly_entries": 3,      # 종목당 월 최대 진입
}


def main():
    loader = RuntimeYFinanceLoader()
    market_filter = MarketFilter(MARKET_CONFIG)
    stock_engine = StockEngine(STOCK_CONFIG, RISK_CONFIG)
    etf_engine = ETFEngine(MARKET_CONFIG, RISK_CONFIG)
    portfolio_manager = PortfolioManager(
        state_file="data/portfolio_state.json",
        portfolio_cfg=PORTFOLIO_CFG,
    )
    reporter = ReportGenerator()

    analysis_date = date.today()
    start_date = analysis_date - timedelta(days=180)

    portfolio_manager.initialize(
        total_capital=100_000.0,
        today=analysis_date,
    )

    qqq_df = loader.load_single(MARKET_CONFIG["benchmark_tickers"]["qqq"], start_date, analysis_date)
    spy_df = loader.load_single(MARKET_CONFIG["benchmark_tickers"]["spy"], start_date, analysis_date)
    vix_df = loader.load_single(MARKET_CONFIG["benchmark_tickers"]["vix"], start_date, analysis_date)

    if qqq_df is None or spy_df is None or vix_df is None:
        print("benchmark data load failed")
        return

    mf_result = market_filter.run(qqq_df, spy_df, vix_df, analysis_date)

    tickers = loader.get_ndx100_components()
    universe_data = loader.load_multiple(tickers, start_date, analysis_date)
    universe_data["QQQ"] = qqq_df
    universe_data["SPY"] = spy_df

    account = portfolio_manager.account
    if account.total_capital == 0:
        account = AccountState(
            total_capital=100_000.0,
            daily_pnl_pct=0.0,
            weekly_pnl_pct=0.0,
            monthly_pnl_pct=0.0,
        )

    weekly_pool, candidates = stock_engine.run(
        universe_data=universe_data,
        benchmark_df=qqq_df,
        market_filter=mf_result,
        account=account,
        analysis_date=analysis_date,
        force_refresh=True,
    )

    rs_map = {entry["ticker"]: entry["rs_score"] for entry in weekly_pool.pool}
    portfolio_manager.sync_rs_scores(rs_map)
    portfolio_manager.update_positions(universe_data)

    etf_candidates = etf_engine.run(
        universe_data=universe_data,
        market_filter=mf_result,
        account=account,
        analysis_date=analysis_date,
    )

    report = reporter.generate(
        market_filter=mf_result,
        stock_candidates=candidates,
        etf_candidates=etf_candidates,
        analysis_date=analysis_date,
        universe_data=universe_data,
        positions=portfolio_manager.portfolio.positions,
    )
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(report)


if __name__ == "__main__":
    main()