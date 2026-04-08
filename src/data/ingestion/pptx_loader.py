import logging
from typing import List, Dict, Any
from unstructured.partition.pptx import partition_pptx
from .base_loader import BaseLoader
from .utils import convert_to_dialogue_format

logger = logging.getLogger(__name__)

class PptxLoader(BaseLoader):
    """
    A loader for PowerPoint PPTX files.
    """
    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads and partitions a PPTX file.
        """
        logger.info(f"Loading PPTX file: {file_path}")
        try:
            elements = partition_pptx(filename=file_path)
            return convert_to_dialogue_format(elements, metadata)
        except Exception as e:
            logger.error(f"Failed to load or parse PPTX file {file_path}: {e}", exc_info=True)
            return []