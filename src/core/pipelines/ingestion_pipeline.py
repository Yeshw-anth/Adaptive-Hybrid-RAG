import logging
from typing import List
from llama_index.core.schema import Document

from src.data.ingestion.loader_factory import get_loader
from src.chunking.chunking_engine import ChunkingEngine
from src.data.embedding.embedder import Embedder
from src.config import settings
from src.core.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    A centralized pipeline for ingesting documents into the system.
    It orchestrates loading, parsing, and chunking of documents.
    """
    def __init__(self):
        self.llm_client = OllamaClient()
        embedding_model = Embedder(model_name=settings.EMBED_MODEL_NAME)
        self.chunking_engine = ChunkingEngine(embedding_model=embedding_model)
        logger.info("IngestionPipeline initialized.")

    async def run(self, file_path: str) -> List[Document]:
        """
        Executes the full ingestion pipeline for a single file.

        Args:
            file_path (str): The path to the document.

        Returns:
            List[Document]: A list of parsed LlamaIndex Document objects (nodes).
        """
        logger.debug(f"Starting ingestion for file: {file_path}")
        try:
            # 1. Select and get the correct loader for the file type
            loader = get_loader(file_path, self.llm_client)
            logger.debug(f"Using loader: {type(loader).__name__} for {file_path}")

            # 2. Load the document to get structured elements
            elements = await loader.load(file_path)
            if not elements:
                logger.warning(f"No elements were extracted from {file_path}. Aborting ingestion.")
                return []
            logger.info(f"Loader extracted {len(elements)} raw elements from document.")

            # Convert raw elements (dicts) into LlamaIndex Document objects
            # The 'text' of the document is the 'content' from our loader.
            # The full dictionary is preserved in the metadata.
            documents = [Document(text=el.get('text', ''), metadata=el) for el in elements]
            logger.debug(f"Converted {len(elements)} raw elements into {len(documents)} LlamaIndex Documents.")

            # 3. Pass the LlamaIndex Documents to the Chunking Engine
            nodes = await self.chunking_engine.chunk_document(documents, file_path)
            logger.info(f"Successfully chunked document into {len(nodes)} nodes.")

            logger.debug("Ingestion pipeline completed successfully.")
            return nodes

        except Exception as e:
            logger.error(f"An error occurred during the ingestion pipeline for {file_path}: {e}", exc_info=True)
            return []