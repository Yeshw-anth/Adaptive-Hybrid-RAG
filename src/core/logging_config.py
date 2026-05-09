import logging
import sys
from loguru import logger
from pathlib import Path
import json

from src.config.settings import settings

class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging messages and redirects them to Loguru.
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# --- Centralized Logging Configuration ---

# Ensure the log directory exists
log_file = settings.LOG_FILE_PATH
log_file.parent.mkdir(parents=True, exist_ok=True)

# Define the configuration for all handlers
handlers = [
    {
        "sink": sys.stdout,
        "level": settings.LOG_LEVEL.upper(),
        "format": settings.LOG_FORMAT_CONSOLE,
        "colorize": True,
        "backtrace": True,
        "diagnose": True,
    },
    {
        "sink": log_file,
        "level": settings.LOG_LEVEL.upper(),
        "format": settings.LOG_FORMAT_FILE,
        "rotation": settings.LOG_ROTATION,
        "retention": settings.LOG_RETENTION,
        "compression": "zip",
        "enqueue": True,  # Make file logging asynchronous
        "backtrace": True,
        "diagnose": True,
    },
]

# Configure the logger in one go. This replaces all existing handlers.
# The patcher ensures that a 'query_id' is always present in the 'extra' dict.
logger.configure(
    handlers=handlers,
    patcher=lambda record: record["extra"].setdefault("query_id", "System"),
)


# Intercept standard logging to redirect logs from other libraries (like huggingface)
# to our configured Loguru sink. This prevents double logging.
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

logger.info(f"Logger configured: Level={settings.LOG_LEVEL}, Directory={settings.LOG_DIR}")

# --- End of Configuration ---