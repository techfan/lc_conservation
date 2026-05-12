import logging
import os
from logging.handlers import TimedRotatingFileHandler

from ddtrace import patch


class TraceLogFormatter(logging.Formatter):
    """
    自定义日志格式：
    时间戳 | 线程名 | trace_id | 级别 | 文件名 | 日志内容
    """
    def __init__(self):
        super().__init__(
            fmt=(
                "%(asctime)s | "
                "%(threadName)s | "
                "%(dd.trace_id)s | "
                "%(dd.span_id)s | "
                "%(levelname)s | "
                "%(filename)s | "
                "%(message)s"
            )
        )

def setup_logger():
    _base_config()
    _ddtrace_config()


def _base_config():
    # 根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清空默认

    # ===================== 1. 控制台输出 =====================
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(TraceLogFormatter())
    logger.addHandler(console_handler)

    # ===================== 2. 文件输出（每月滚动） =====================
    log_path = os.path.join(os.getcwd(), "logs")
    log_file = os.path.join(log_path, "app.log")
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="M",  # 按月回滚
        interval=1,  # 1个月一次
        backupCount=12,  # 最多保留12个月日志
        encoding="utf-8",
        delay=True
    )
    file_handler.setFormatter(TraceLogFormatter())
    logger.addHandler(file_handler)


def _ddtrace_config():
    os.environ["DD_CONTEXT_THREADING"] = "true"
    patch(fastapi=True, logging = True, requests =  True)