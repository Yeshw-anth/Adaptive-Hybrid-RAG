import logging
from typing import List, Dict, Any
from unstructured.partition.json import partition_json
from .base_loader import BaseLoader
from .utils import convert_to_dialogue_format

logger = logging.getLogger(__name__)

class JsonLoader(BaseLoader):
    """
    A loader for JSON files.
    """
    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads and partitions a JSON file.
        """
        logger.info(f"Loading JSON file: {file_path}")
        try:
            elements = partition_json(filename=file_path)
            return convert_to_dialogue_format(elements, metadata)
        except Exception as e:
            logger.error(f"Failed to load or parse JSON file {file_path}: {e}", exc_info=True)
            return []