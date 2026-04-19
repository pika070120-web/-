import schedule
import time
import subprocess
import sys
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def run_main():
    now = datetime.now()
    # 주말 제외
    if now.weekday() >= 5:
        logger.info("주말 - 스킵")
        return
    logger.info("메인 러너 시작")
    result = subprocess.run(
        [sys.executable, "-m", "runners.main_runner"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode == 0:
        logger.info("완료")
    else:
        logger.error(f"오류: {result.stderr[:200]}")


# 매일 미국 장 마감 후 한국 시간 기준 07:00 (ET 17:00)
schedule.every().day.at("07:00").do(run_main)

if __name__ == "__main__":
    logger.info("스케줄러 시작 - 매일 07:00 실행")
    run_main()  # 시작 시 즉시 1회 실행
    while True:
        schedule.run_pending()
        time.sleep(60)