from abc import ABC, abstractmethod
from typing import List, Dict, Any
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding

class BaseChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.

    Strategies can implement either `process_document` for whole-file processing
    or `process_section` for section-based processing.
    """
    def __init__(self, embedding_model: BaseEmbedding = None):
        """
        Initializes the strategy.
        
        Args:
            embedding_model: An optional embedding model, required for semantic chunking.
        """
        self.embedding_model = embedding_model

    @abstractmethod
    async def process(self, elements: List[Any], file_path: str) -> List[TextNode]:
        """
        Processes a list of elements from a document or a section of a document.

        Args:
            elements (List[Any]): The list of elements (e.g., from a loader or a section).
            file_path (str): The path to the original file.

        Returns:
            List[TextNode]: A list of chunked nodes.
        """
        pass