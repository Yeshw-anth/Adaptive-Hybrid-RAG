from typing import List, Dict, Any
import os

from .base_loader import BaseLoader

class TextLoader(BaseLoader):
    """
    A loader for .txt files.
    """
    def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads a .txt file and returns its content with metadata.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        file_name = os.path.basename(file_path)
        doc_metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_type": ".txt",
            "chunk_id": f"{file_name}-0"
        }
            
        return [{
            "text": text,
            "metadata": doc_metadata
        }]