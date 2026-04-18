# backtest/backtest_runner.py  (v1.0)
"""
백테스트 실행 스크립트
실행: python -m backtest.backtest_runner
"""

import sys
import logging
from datetime import date, timedelta

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING)

from core.config_loader import MARKET_CONFIG, RISK_CONFIG, STOCK_CONFIG
from data.runtime_yf_loader import RuntimeYFinanceLoader
from backtest.backtest_engine import BacktestEngine


def print_summary(summary) -> None:
    L = []
    L.append("=" * 60)
    L.append("  백테스트 결과")
    L.append("=" * 60)
    L.append(f"  기간        : {summary.start_date} ~ {summary.end_date}")
    L.append(f"  초기 자본   : ${summary.initial_capital:,.0f}")
    L.append(f"  최종 자산   : ${summary.final_value:,.0f}")
    sign = "+" if summary.total_return_pct >= 0 else ""
    L.append(f"  총 수익률   : {sign}{summary.total_return_pct:.2f}%")
    L.append(f"  최대 낙폭   : -{summary.max_drawdown_pct:.2f}%")
    L.append(f"  총 거래 수  : {summary.total_trades}회")
    L.append(f"  승률        : {summary.win_rate_pct:.1f}%  "
              f"(승 {summary.winning_trades} / 패 {summary.losing_trades})")
    L.append(f"  평균 수익   : +{summary.avg_win_pct:.2f}%")
    L.append(f"  평균 손실   : {summary.avg_loss_pct:.2f}%")
    L.append("")
    L.append("━━ 거래 내역 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sells = [t for t in summary.trade_log if t["action"] == "SELL"]
    for t in sells:
        sign = "+" if t.get("pnl_pct", 0) >= 0 else ""
        L.append(
            f"  {t['date']}  {t['ticker']:<6}  "
            f"{sign}{t.get('pnl_pct', 0):.2f}%  "
            f"보유{t.get('hold_days', 0)}일  [{t.get('reason', '')}]"
        )
    L.append("=" * 60)
    print("\n".join(L))


def main():
    # ── 기간 설정 ───────────────────────────
    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)   # 1년 백테스트
    data_start = start_date - timedelta(days=120) # 지표 계산용 여유

    print(f"데이터 로딩 중... ({data_start} ~ {end_date})")

    loader = RuntimeYFinanceLoader()

    # 벤치마크
    qqq_df = loader.load_single("QQQ", data_start, end_date)
    spy_df = loader.load_single("SPY", data_start, end_date)
    vix_df = loader.load_single("VIX", data_start, end_date)

    if qqq_df is None or spy_df is None or vix_df is None:
        print("벤치마크 데이터 로드 실패")
        return

    # 유니버스 (나스닥100)
    tickers = loader.get_ndx100_components()
    print(f"종목 데이터 로딩 중... ({len(tickers)}종목)")
    universe_data = loader.load_multiple(tickers, data_start, end_date)
    universe_data["QQQ"] = qqq_df
    universe_data["SPY"] = spy_df
    universe_data["VIX"] = vix_df

    print(f"로드 완료: {len(universe_data)}개\n백테스트 실행 중...")

    engine = BacktestEngine(
        market_cfg=MARKET_CONFIG,
        stock_cfg=STOCK_CONFIG,
        risk_cfg=RISK_CONFIG,
        initial_capital=100_000.0,
    )

    summary = engine.run(
        full_data=universe_data,
        start_date=start_date,
        end_date=end_date,
    )

    print_summary(summary)


if __name__ == "__main__":
    main()