
import os
import hashlib
from typing import Dict, Any
from src.data.schemas import DocumentMetadata

class DataProfiler:
    """
    Analyzes a document to extract rich metadata.
    This metadata is used by the StrategySelector to make intelligent decisions.
    """

    def profile(self, file_path: str, file_content: bytes) -> DocumentMetadata:
        """
        Generates a metadata profile for a given document.

        Args:
            file_path (str): The path to the document.
            file_content (bytes): The raw content of the document.

        Returns:
            DocumentMetadata: A Pydantic model containing the extracted metadata.
        """
        file_stats = os.stat(file_path)
        
        # Create a dictionary with all the metadata
        metadata_dict = {
            "file_name": os.path.basename(file_path),
            "word_count": len(file_content.decode('utf-8', errors='ignore').split()),
            "content_hash": self._calculate_hash(file_content),
            # You can add other fields required by other components here
            "file_type": os.path.splitext(file_path)[1].lower(),
            "file_size": file_stats.st_size,
        }
        
        # Create the Pydantic model, ensuring all required fields are present
        return DocumentMetadata(**metadata_dict)

    def _calculate_hash(self, content: bytes) -> str:
        """Calculates the SHA-256 hash of the file content."""
        return hashlib.sha256(content).hexdigest()