import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import settings

def setup_logging():
    """
    Configures centralized logging for the entire application.
    - Logs to both a file and the console.
    - Uses a rotating file handler to manage log file size.
    """
    log_level = settings.LOG_LEVEL
    log_file = settings.LOG_FILE_PATH

    # Create log directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)

    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # --- File Handler ---
    # This handler will overwrite the log file on each run (filemode='w').
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(name)-30s: %(levelname)-8s %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Add handlers to the root logger
    # Avoid adding handlers if they already exist (e.g., during hot reloads)
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    # --- Suppress Noisy Third-Party Loggers ---
    # Set the logging level for 'pdfminer' to WARNING to avoid excessive debug output.
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured. Level: {log_level}. Output file: {log_file}")