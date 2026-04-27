import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

class AlpacaBroker:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = "https://paper-api.alpaca.markets"
        self._api = None

    def connect(self) -> bool:
        try:
            import alpaca_trade_api as tradeapi
            self._api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                self.base_url,
                api_version="v2",
            )
            account = self._api.get_account()
            print(f"Alpaca 연결 성공 - 잔고: ${float(account.cash):,.0f}")
            return True
        except Exception as e:
            print(f"Alpaca 연결 실패: {e}")
            return False

    def get_account_info(self) -> Optional[dict]:
        if self._api is None:
            return None
        try:
            acc = self._api.get_account()
            return {
                "cash": float(acc.cash),
                "portfolio_value": float(acc.portfolio_value),
                "buying_power": float(acc.buying_power),
            }
        except Exception as e:
            print(f"계좌 조회 실패: {e}")
            return None