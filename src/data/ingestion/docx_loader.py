import logging
from typing import List, Dict, Any
from unstructured.partition.docx import partition_docx
from .base_loader import BaseLoader
from .utils import convert_to_dialogue_format

logger = logging.getLogger(__name__)

class DocxLoader(BaseLoader):
    """
    A loader for DOCX files.
    """
    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads and partitions a DOCX file.
        """
        logger.info(f"Loading DOCX file: {file_path}")
        try:
            elements = partition_docx(filename=file_path)
            return convert_to_dialogue_format(elements, metadata)
        except Exception as e:
            logger.error(f"Failed to load or parse DOCX file {file_path}: {e}", exc_info=True)
            return []