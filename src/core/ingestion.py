# src/core/ingestion.py
import logging
from typing import List, Dict, Type, Protocol
from llama_index.core.schema import TextNode
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from src.chunking.chunking_engine import ChunkingEngine
from src.data.ingestion.pdf_router import IntelligentPDFRouter

logger = logging.getLogger(__name__)


from src.data.ingestion.base_loader import Loader, LoaderType


class UnstructuredLoader(Loader):
    """A standard loader for all file types supported by the `unstructured` library, except for PDFs."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def process(self) -> List[Element]:
        logger.info(f"Using standard UnstructuredLoader for: {self.file_path}")
        return partition(filename=self.file_path, strategy="auto")


class IngestionRouter:
    """Dynamically selects the correct loader based on the file extension."""

    # Map file extensions to their corresponding loader class
    _loader_map: Dict[str, LoaderType] = {
        ".pdf": IntelligentPDFRouter,
        # Add other specific loaders here if needed in the future
    }
    
    # A set of extensions that the default loader will handle
    _default_extensions = {".docx", ".doc", ".txt", ".md", ".html", ".htm", ".json", ".csv"}

    @classmethod
    def get_loader(cls, file_path: str) -> Loader:
        """
        Returns an instance of the appropriate loader for the given file path.
        """
        file_ext = f".{file_path.split('.')[-1].lower()}"
        
        # 1. Check for a specific, high-priority loader (like for PDFs)
        loader_class = cls._loader_map.get(file_ext)
        if loader_class:
            logger.info(f"Found specific loader '{loader_class.__name__}' for file type '{file_ext}'.")
            return loader_class(file_path=file_path)

        # 2. Check if the file type is supported by the default loader
        if file_ext in cls._default_extensions:
            logger.info(f"Using default 'UnstructuredLoader' for file type '{file_ext}'.")
            return UnstructuredLoader(file_path=file_path)
            
        # 3. If no suitable loader is found, raise an error
        raise ValueError(f"No suitable loader found for file type '{file_ext}'")


class IngestionPipeline:
    """
    A unified class to orchestrate the document ingestion process. It uses an
    IngestionRouter to dynamically select the correct parsing strategy for each file.
    """

    def __init__(self, chunking_engine: ChunkingEngine):
        self.chunking_engine = chunking_engine

    async def ingest_file(self, file_path: str) -> List[TextNode]:
        """
        Ingests a single file using the dynamically selected best loader.
        """
        logger.info(f"Starting unified ingestion for: {file_path}")
        try:
            # 1. Dynamically select and instantiate the correct loader
            loader = IngestionRouter.get_loader(file_path)
            
            # 2. Process the file to get a list of unstructured.Elements
            elements = loader.process()
            logger.info(f"Successfully partitioned file {file_path} into {len(elements)} elements.")

            # 3. Use the ChunkingEngine to convert elements into TextNodes
            nodes = await self.chunking_engine.chunk_document(elements, file_path)
            
            logger.info(f"Unified ingestion successful for {file_path}. Generated {len(nodes)} nodes.")
            return nodes
        except Exception as e:
            logger.error(f"Error during unified ingestion for {file_path}: {e}", exc_info=True)
            return []