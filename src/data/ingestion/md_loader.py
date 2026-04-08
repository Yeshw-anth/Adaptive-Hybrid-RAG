import logging
from typing import List, Dict, Any
from unstructured.partition.md import partition_md
from .base_loader import BaseLoader
from .utils import convert_to_dialogue_format

logger = logging.getLogger(__name__)

class MdLoader(BaseLoader):
    """
    A loader for Markdown files.
    Uses the unstructured library to partition the document.
    """
    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads and partitions a Markdown file into structured elements.

        Args:
            file_path: The path to the Markdown file.
            metadata: Optional metadata to attach to the elements.

        Returns:
            A list of dictionaries, where each dictionary represents a
            structured element from the document.
        """
        logger.info(f"Loading Markdown file: {file_path}")
        try:
            elements = partition_md(filename=file_path)
            return convert_to_dialogue_format(elements, metadata)
        except Exception as e:
            logger.error(f"Failed to load or parse Markdown file {file_path}: {e}", exc_info=True)
            return []