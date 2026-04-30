# src/core/ingestion.py
from src.core.logging_config import logger
from typing import List, Dict, Type, Protocol
from llama_index.core.schema import TextNode
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from src.config.settings import settings
from src.chunking.chunking_engine import ChunkingEngine
from src.data.ingestion.pdf_router import IntelligentPDFRouter
from src.core.graph.graph_builder import KnowledgeGraphBuilder
from src.core.graph.graph_store import get_graph_store
from src.core.llm.ollama_client import OllamaClient


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


import asyncio
import math

# ... (other imports)

TARGET_BATCH_COUNT = 10  # Aim for this many concurrent batches
MAX_BATCH_SIZE = 30      # But never exceed this many nodes in a single batch

class IngestionPipeline:
    """
    A unified class to orchestrate the document ingestion process. It uses an
    IngestionRouter to dynamically select the correct parsing strategy for each file.
    """

    def __init__(self, chunking_engine: ChunkingEngine, graph_builder: KnowledgeGraphBuilder, llm_client: OllamaClient):
        self.chunking_engine = chunking_engine
        self.graph_builder = graph_builder
        self.graph_store = get_graph_store()
        self.llm_client = llm_client

    def _normalize_entity(self, entity: str) -> str:
        """
        A simple function to normalize entity names for consistency in the graph.
        - Converts to lowercase
        - Removes common articles
        - Strips leading/trailing whitespace
        """
        return entity.lower().replace("the ", "").replace("a ", "").replace("an ", "").strip()

    async def ingest_file(self, file_path: str) -> List[TextNode]:
        """
        Ingests a single file, dynamically calculating batch size for efficient processing.
        """
        logger.info(f"Starting unified ingestion for: {file_path}")
        try:
            # 1. & 2. Load and partition the file
            loader = IngestionRouter.get_loader(file_path)
            elements = loader.process()
            logger.info(f"Successfully partitioned file {file_path} into {len(elements)} elements.")

            # 3. Chunk the document into TextNodes
            nodes = await self.chunking_engine.chunk_document(elements, file_path)
            if not nodes:
                logger.warning(f"No nodes were generated for {file_path}. Aborting ingestion.")
                return []
            logger.info(f"Chunked document into {len(nodes)} nodes.")

            # 4. Dynamically calculate batch size for concurrency, respecting the max size
            num_nodes = len(nodes)

            # Aim for TARGET_BATCH_COUNT batches, but cap the size of each batch
            ideal_batch_size = math.ceil(num_nodes / TARGET_BATCH_COUNT)
            batch_size = min(MAX_BATCH_SIZE, ideal_batch_size)

            if batch_size == 0:
                logger.warning("Batch size calculated to zero, no nodes to process.")
                return nodes

            node_batches = [nodes[i:i + batch_size] for i in range(0, num_nodes, batch_size)]
            num_batches = len(node_batches)
            logger.info(f"Processing {num_nodes} nodes in {num_batches} batches of up to {batch_size} nodes each.")

            # 5. Create and run concurrent tasks
            tasks = []
            for i, batch in enumerate(node_batches):
                texts_to_process = [node.get_content() for node in batch if node.get_content()]
                if texts_to_process:
                    task = self.graph_builder.generate_triples_batch(texts_to_process, settings.DEFAULT_LLM_MODEL)
                    tasks.append(task)

            # 6. Process batches as they complete for better observability
            total_triples = 0
            if tasks:
                completed_batches = 0
                for future in asyncio.as_completed(tasks):
                    try:
                        result = await future
                        completed_batches += 1
                        if result:
                            logger.info(f"Batch {completed_batches}/{len(tasks)} complete. Adding {len(result)} triples to the graph.")
                            for subj, pred, obj in result:
                                # Normalize entities before adding to the graph
                                norm_subj = self._normalize_entity(subj)
                                norm_obj = self._normalize_entity(obj)
                                
                                # Add normalized nodes and the original edge
                                self.graph_store.add_node(norm_subj)
                                self.graph_store.add_node(norm_obj)
                                self.graph_store.add_edge(norm_subj, norm_obj, label=pred)
                            total_triples += len(result)
                        else:
                            logger.info(f"Batch {completed_batches}/{len(tasks)} complete. No triples extracted.")
                    except Exception as e:
                        logger.error(f"A batch task failed during triple extraction: {e}", exc_info=True)

            # 7. Save the graph once after all additions
            if total_triples > 0:
                if self.graph_store.save_graph():
                    logger.info(f"Graph saved successfully. Total triples added: {total_triples}")
                else:
                    logger.error(f"Failed to save graph. Total triples processed but not persisted: {total_triples}")

            logger.info(f"Unified ingestion successful for {file_path}. Generated {len(nodes)} nodes and {total_triples} triples.")
            return nodes
        except Exception as e:
            logger.error(f"Error during unified ingestion for {file_path}: {e}", exc_info=True)
            raise