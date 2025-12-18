import logging
import logging.config
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": (
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - " "%(funcName)s(): %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "[%(levelname)s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "console_debug": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "file_error": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "error.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file", "file_error"],
        },
        "database": {
            "level": "DEBUG",
            "handlers": ["console", "file", "file_error"],
            "propagate": False,
        },
        "inference_utils": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "main": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "app": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

_logging_configured = False


def setup_logging():
    global _logging_configured
    if _logging_configured:
        return logging.getLogger(__name__)

    # Ensure logs directory exists BEFORE configuring
    LOGS_DIR.mkdir(exist_ok=True)

    logging.config.dictConfig(LOGGING_CONFIG)
    _logging_configured = True
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")

    return logger
