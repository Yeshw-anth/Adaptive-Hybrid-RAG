from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document

class IngestionStrategy(ABC):
    """Abstract base class for document ingestion strategies."""

    @abstractmethod
    def ingest(self, file_path: str) -> List[Document]:
        """
        Ingests a document from a given path.

        Args:
            file_path (str): The path to the document to ingest.

        Returns:
            List[Document]: A list of LlamaIndex Document objects.
        """
        pass