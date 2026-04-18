# backtest/backtest_runner.py  (v1.3 - 1년 전체 백테스트)
import sys
import logging
from collections import Counter
from datetime import date, timedelta

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING)

from core.config_loader import MARKET_CONFIG, RISK_CONFIG, STOCK_CONFIG
from data.runtime_yf_loader import RuntimeYFinanceLoader
from backtest.backtest_engine import BacktestEngine


def print_summary(summary) -> None:
    L = []
    L.append("=" * 60)
    L.append("  백테스트 결과 (Pullback + Breakout 전략)")
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
    L.append("━━ 시장 상태 분포 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    state_cnt = Counter(d.market_state for d in summary.daily_results)
    total_days = len(summary.daily_results)
    for state, cnt in sorted(state_cnt.items()):
        L.append(f"  {state:<12}: {cnt:>3}일 ({cnt/total_days*100:.0f}%)")

    L.append("")
    L.append("━━ 신호 발생 현황 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    entry_days = [d for d in summary.daily_results if d.new_entries]
    all_entries = []
    for d in summary.daily_results:
        all_entries.extend(d.new_entries)
    ticker_cnt = Counter(all_entries)
    L.append(f"  신호 발생일 : {len(entry_days)}일 / {total_days}일")
    L.append(f"  총 진입 시도: {len(all_entries)}회")
    if ticker_cnt:
        L.append("  종목별 진입 (상위10):")
        for ticker, cnt in ticker_cnt.most_common(10):
            L.append(f"    {ticker:<6}: {cnt}회")

    L.append("")
    L.append("━━ 거래 내역 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sells = [t for t in summary.trade_log if t["action"] == "SELL"]
    if not sells:
        L.append("  청산 거래 없음")
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
    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=550)   # 약 1.5년
    data_start = start_date - timedelta(days=120)

    print(f"데이터 로딩 중... ({data_start} ~ {end_date})")

    loader = RuntimeYFinanceLoader()
    qqq_df = loader.load_single("QQQ", data_start, end_date)
    spy_df = loader.load_single("SPY", data_start, end_date)
    vix_df = loader.load_single("VIX", data_start, end_date)

    if qqq_df is None or spy_df is None or vix_df is None:
        print("벤치마크 데이터 로드 실패")
        return

    tickers = loader.get_ndx100_components()
    print(f"종목 데이터 로딩 중... ({len(tickers)}종목)")
    universe_data = loader.load_multiple(tickers, data_start, end_date)
    universe_data["QQQ"] = qqq_df
    universe_data["SPY"] = spy_df
    universe_data["VIX"] = vix_df

    print(f"로드 완료: {len(universe_data)}개\n백테스트 실행 중... (3~5분 소요)")

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