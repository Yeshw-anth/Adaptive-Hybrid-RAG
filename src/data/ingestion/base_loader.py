
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from llama_index.core.schema import TextNode

class BaseLoader(ABC):
    """
    An abstract base class for document loaders. It defines a standard interface
    for loading documents and converting them into LlamaIndex TextNode objects.
    """

    @abstractmethod
    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads a document from a file and returns a list of raw element dictionaries.
        This is the method that subclasses must implement.
        """
        pass

    async def load_and_chunk(self, file_path: str, metadata: Dict[str, Any] = None) -> List[TextNode]:
        """
        Loads a document and converts its elements into a list of TextNode objects.
        This provides a consistent output for the ingestion pipeline.
        """
        elements = await self.load(file_path, metadata)
        
        nodes = []
        for element in elements:
            if not isinstance(element, dict) or 'text' not in element:
                continue

            node = TextNode(
                text=element.get('text'),
                metadata=element.get('metadata', {})
            )
            nodes.append(node)
            
        return nodes