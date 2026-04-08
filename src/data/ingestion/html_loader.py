import logging
from typing import List, Dict, Any
from unstructured.partition.html import partition_html
from .base_loader import BaseLoader
from .utils import convert_to_dialogue_format

logger = logging.getLogger(__name__)

class HtmlLoader(BaseLoader):
    """
    A loader for HTML files.
    """
    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads and partitions an HTML file.
        """
        logger.info(f"Loading HTML file: {file_path}")
        try:
            elements = partition_html(filename=file_path)
            return convert_to_dialogue_format(elements, metadata)
        except Exception as e:
            logger.error(f"Failed to load or parse HTML file {file_path}: {e}", exc_info=True)
            return []