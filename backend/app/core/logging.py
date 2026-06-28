import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(log_level: str = "INFO") -> None:
    # 統一後端 log 格式，方便日後追蹤 API、Agent、ML 任務流程。
    normalized_level = log_level.upper()

    logging.basicConfig(
        level=normalized_level,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
