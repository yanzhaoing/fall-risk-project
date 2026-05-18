"""日志配置"""
import logging
import sys
from pathlib import Path
from datetime import datetime

from config.settings import CFG_PATHS


def setup_logger(
    name: str = "fall_risk",
    level: str = "INFO",
    log_dir: str = str(CFG_PATHS.LOGS_DIR),
) -> logging.Logger:
    """
    配置日志器

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        # 控制台
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(ch)

        # 文件
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_path / f"{name}_{datetime.now():%Y%m%d}.log"
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(fh)

    return logger
