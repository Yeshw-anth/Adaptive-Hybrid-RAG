import logging
from typing import List, Dict, Any
from unstructured.partition.csv import partition_csv
from .base_loader import BaseLoader
from .utils import convert_to_dialogue_format

logger = logging.getLogger(__name__)

class CsvLoader(BaseLoader):
    """
    A loader for CSV files.
    """
    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads and partitions a CSV file.
        """
        logger.info(f"Loading CSV file: {file_path}")
        try:
            elements = partition_csv(filename=file_path)
            return convert_to_dialogue_format(elements, metadata)
        except Exception as e:
            logger.error(f"Failed to load or parse CSV file {file_path}: {e}", exc_info=True)
            return []