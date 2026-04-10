# src/core/ingestion.py
import logging
from typing import List
from llama_index.core.schema import TextNode
from src.chunking.chunking_engine import ChunkingEngine
from src.data.ingestion.loader_factory import get_loader
from src.core.llm.ollama_client import OllamaClient
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    A class to orchestrate the document ingestion process, supporting multiple strategies.
    """

    def __init__(self, chunking_engine: ChunkingEngine, llm_client: OllamaClient):
        self.chunking_engine = chunking_engine
        self.llm_client = llm_client

    async def ingest_file_unstructured(self, file_path: str) -> List[TextNode]:
        """
        Ingests a file using the modern, 'unstructured'-based ChunkingEngine.
        This is the recommended approach.
        """
        logger.info(f"Starting 'unstructured' ingestion for: {file_path}")
        try:
            # 1. Partition the file to get raw elements
            elements = partition(filename=file_path, strategy="auto")
            logger.info(f"Successfully partitioned file {file_path} into {len(elements)} elements.")

            # 2. Use the refactored ChunkingEngine to get nodes directly
            nodes = await self.chunking_engine.chunk_document(elements, file_path)
            
            logger.info(f"'unstructured' ingestion successful for {file_path}. Generated {len(nodes)} nodes.")
            return nodes
        except Exception as e:
            logger.error(f"Error during 'unstructured' ingestion for {file_path}: {e}", exc_info=True)
            return []

    async def ingest_file_manual(self, file_path: str) -> List[TextNode]:
        """
        Ingests a file using the legacy, 'manual' loader-based approach.
        This is kept for benchmarking and fallback purposes.
        """
        logger.info(f"Starting 'manual' ingestion for: {file_path}")
        try:
            # 1. Select the correct loader using the old factory
            loader = get_loader(file_path, self.llm_client)
            
            # 2. Load the document into simple nodes/chunks
            nodes = await loader.load_and_chunk(file_path=file_path)
            
            logger.info(f"'manual' ingestion successful for {file_path}. Generated {len(nodes)} nodes.")
            return nodes
        except Exception as e:
            logger.error(f"Error during 'manual' ingestion for {file_path}: {e}", exc_info=True)
            return []