from src.core.logging_config import logger
import json
from datetime import datetime
from src.data.schemas import OutputLog
from src.config.settings import settings

class OutputLogger:
    """
    A dedicated logger for capturing outputlogs data.
    """
    def __init__(self, log_file_path: str = None):
        self.log_file_path = log_file_path or settings.OUTPUT_LOG_FILE
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

    def log(self, output_log: OutputLog):
        """
        Appends a structured output log to the specified log file.

        Args:
            eval_log: An EvaluationLog object containing the data to log.
        """
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                # Convert Pydantic model to dict and then to a JSON string
                log_entry = output_log.model_dump_json()
                f.write(log_entry + "\n")
        except Exception as e:
            logger.error(f"Failed to write to output log: {e}")