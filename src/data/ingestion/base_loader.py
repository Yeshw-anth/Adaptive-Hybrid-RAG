
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLoader(ABC):
    """
    An abstract base class for document loaders.
    """

    @abstractmethod
    def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads a document from a file and returns a list of document objects.

        Args:
            file_path (str): The path to the file to be loaded.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a document with 'text' and 'metadata'.
        """
        pass