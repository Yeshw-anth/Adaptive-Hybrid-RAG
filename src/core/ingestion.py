# src/core/ingestion.py
from src.core.logging_config import logger
from typing import List, Dict, Type, Protocol
from llama_index.core.schema import TextNode
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.config.settings import settings
from src.chunking.chunking_engine import ChunkingEngine
from src.data.ingestion.pdf_router import IntelligentPDFRouter
from src.core.graph.graph_builder import KnowledgeGraphBuilder
from src.core.graph.graph_store import  GraphStore
from src.core.llm.ollama_client import OllamaClient


from src.data.ingestion.base_loader import Loader, LoaderType


import asyncio
import math
import os
import hashlib

from llama_index.core import VectorStoreIndex

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except Exception as e:
    logger.error(f"An error occurred during NLTK data check/download: {e}")
    # Attempt to download all essential packages, as we can't be sure which one failed.
    logger.info("Attempting to download all required NLTK packages...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK packages downloaded successfully.")
    except Exception as download_e:
        logger.critical(f"Failed to download essential NLTK packages: {download_e}. The application may not function correctly.")
        # Depending on the application's requirements, you might want to raise the exception here to halt execution.
        # raise download_e

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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



def get_file_hash(file_path: str) -> str:
    """Computes the SHA-256 hash of a file's content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# ... (other imports)

TARGET_BATCH_COUNT = 10  # Aim for this many concurrent batches
MAX_BATCH_SIZE = 30      # But never exceed this many nodes in a single batch

class IngestionPipeline:
    """
    A unified class to orchestrate the document ingestion process. It uses an
    IngestionRouter to dynamically select the correct parsing strategy for each file.
    """

    def __init__(self, chunking_engine: ChunkingEngine, graph_builder: KnowledgeGraphBuilder, llm_client: OllamaClient, graph_store: "GraphStore", vector_index: VectorStoreIndex):
        self.chunking_engine = chunking_engine
        self.graph_builder = graph_builder
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.vector_index = vector_index

    def _normalize_entity(self, entity: str) -> str:
        """
        Normalizes entity names for consistency in the graph using lemmatization.
        - Converts to lowercase
        - Tokenizes
        - Lemmatizes each token
        - Removes stop words (except for very short entities)
        - Strips leading/trailing whitespace
        """
        entity = entity.lower().strip()
        tokens = word_tokenize(entity)
        
        # Lemmatize tokens
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stop words, but keep them if the entity is just a stop word (e.g., "IT")
        if len(lemmatized_tokens) > 1:
            lemmatized_tokens = [token for token in lemmatized_tokens if token not in stop_words]
        
        # Re-join and strip
        normalized_entity = ' '.join(lemmatized_tokens).strip()
        
        # If normalization results in an empty string (e.g., entity was only stop words),
        # return the original cleaned entity.
        return normalized_entity if normalized_entity else entity

    async def _build_graph_from_nodes(self, nodes: List[TextNode], filename: str) -> int:
        """Helper to build graph triples from a list of nodes."""
        logger.info(f"Starting graph building process for '{filename}' from {len(nodes)} nodes.")
        total_triples = 0
        num_nodes = len(nodes)
        ideal_batch_size = math.ceil(num_nodes / TARGET_BATCH_COUNT)
        batch_size = min(MAX_BATCH_SIZE, ideal_batch_size)

        if batch_size > 0:
            node_batches = [nodes[i:i + batch_size] for i in range(0, num_nodes, batch_size)]
            num_batches = len(node_batches)
            logger.info(f"Processing {num_nodes} nodes for graph building in {num_batches} batches of up to {batch_size} nodes each.")

            tasks = [
                self.graph_builder.generate_triples_batch(
                    [node.get_content() for node in batch if node.get_content()],
                    settings.DEFAULT_LLM_MODEL
                )
                for batch in node_batches if any(node.get_content() for node in batch)
            ]

            if tasks:
                completed_batches = 0
                for future in asyncio.as_completed(tasks):
                    try:
                        result = await future
                        completed_batches += 1
                        if result:
                            logger.info(f"Graph batch {completed_batches}/{len(tasks)} complete. Adding {len(result)} triples.")
                            for subj, pred, obj in result:
                                norm_subj = self._normalize_entity(subj)
                                norm_obj = self._normalize_entity(obj)
                                self.graph_store.add_node(norm_subj)
                                self.graph_store.add_node(norm_obj)
                                self.graph_store.add_edge(norm_subj, norm_obj, label=pred)
                            total_triples += len(result)
                        else:
                            logger.info(f"Graph batch {completed_batches}/{len(tasks)} complete. No triples extracted.")
                    except Exception as e:
                        logger.error(f"A graph batch task failed during triple extraction: {e}", exc_info=True)
        else:
            logger.warning("Batch size calculated to zero, no nodes to process for graph building.")
        
        logger.info(f"Graph building process for '{filename}' complete. Total triples processed: {total_triples}")
        return total_triples

    async def ingest_file(self, file_path: str) -> List[TextNode]:
        """
        Ingests a single file with a fully robust, four-quadrant check to ensure
        both vector and graph stores are consistent and healthy.
        """
        filename = os.path.basename(file_path)
        file_hash = get_file_hash(file_path)
        logger.info(f"Starting robust ingestion check for: {filename} (hash: {file_hash[:8]}...)")

        # 1. Check the state of both stores
        is_in_graph = self.graph_store.has_source_document(filename, file_hash)
        
        # Manually check if the file exists in the vector store
        is_in_vector = any(
            doc.metadata.get("file_path") == file_path
            for doc in self.vector_index.docstore.docs.values()
        )
        vector_nodes = [
            doc for doc in self.vector_index.docstore.docs.values()
            if doc.metadata.get("file_path") == file_path
        ]

        # --- Case 1: Healthy state (Graph: Good, Vector: Good) ---
        if is_in_graph and is_in_vector:
            logger.info(f"File '{filename}' is fully ingested and verified. Skipping.")
            return []

        # --- Case 2: Vector-only repair (Graph: Good, Vector: Bad) ---
        if is_in_graph and not is_in_vector:
            logger.warning(f"Repairing '{filename}': Graph record exists but vector data is missing. Rebuilding vector store only.")
            try:
                loader = IngestionRouter.get_loader(file_path)
                elements = loader.process()
                nodes = await self.chunking_engine.chunk_document(elements, file_path)
                if not nodes:
                    logger.warning(f"No nodes generated during vector repair for {filename}. Aborting.")
                    return []
                self.vector_index.insert_nodes(nodes)
                logger.info(f"Vector store repair successful for {filename}.")
                return nodes
            except Exception as e:
                logger.error(f"Error during vector store repair for {filename}: {e}", exc_info=True)
                raise

        # --- Case 3: Graph-only repair (Graph: Bad, Vector: Good) ---
        if not is_in_graph and is_in_vector:
            logger.warning(f"Repairing '{filename}': Vector data exists but graph record is missing. Rebuilding graph store only.")
            try:
                nodes_for_graph = vector_nodes
                await self._build_graph_from_nodes(nodes_for_graph, filename)
                
                # Mark document as processed and save the graph
                self.graph_store.add_source_document(filename, file_hash)
                save_future = self.graph_store.save_graph()
                
                def log_save_result(future):
                    try:
                        if future.result(): logger.info(f"Background graph save successful for '{filename}'.")
                        else: logger.error(f"Background graph save failed for '{filename}'.")
                    except Exception as e:
                        logger.error(f"Exception in background graph save for '{filename}': {e}", exc_info=True)
                save_future.add_done_callback(log_save_result)
                
                logger.info(f"Graph store repair successful for {filename}.")
                return nodes_for_graph
            except Exception as e:
                logger.error(f"Error during graph store repair for {filename}: {e}", exc_info=True)
                raise

        # --- Case 4: Full ingestion for new file (Graph: Bad, Vector: Bad) ---
        if not is_in_graph and not is_in_vector:
            logger.info(f"Performing full ingestion for new file: {filename}")
            try:
                # Load, partition, and chunk the document
                loader = IngestionRouter.get_loader(file_path)
                elements = loader.process()
                nodes = await self.chunking_engine.chunk_document(elements, file_path)
                if not nodes:
                    logger.warning(f"No nodes were generated for {filename}. Aborting.")
                    return []

                # Insert into vector store
                self.vector_index.insert_nodes(nodes)
                logger.info(f"Vector store populated for new file '{filename}'.")

                # Build the knowledge graph
                await self._build_graph_from_nodes(nodes, filename)

                # Mark document as processed and save the graph
                self.graph_store.add_source_document(filename, file_hash)
                save_future = self.graph_store.save_graph()
                
                def log_save_result(future):
                    try:
                        if future.result(): logger.info(f"Background graph save successful for '{filename}'.")
                        else: logger.error(f"Background graph save failed for '{filename}'.")
                    except Exception as e:
                        logger.error(f"Exception in background graph save for '{filename}': {e}", exc_info=True)
                save_future.add_done_callback(log_save_result)

                logger.info(f"Full ingestion successful for {filename}.")
                return nodes
            except Exception as e:
                logger.error(f"Error during full ingestion for {filename}: {e}", exc_info=True)
                raise
        
        # This case should ideally not be reached with the logic above.
        logger.critical(f"Reached an unexpected state in ingestion logic for '{filename}'. is_in_graph={is_in_graph}, is_in_vector={is_in_vector}")
        return []