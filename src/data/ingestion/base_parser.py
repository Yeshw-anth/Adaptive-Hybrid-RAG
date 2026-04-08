import abc
from typing import List, Dict, Any
from llama_index.core.schema import TextNode

class BaseParser(abc.ABC):
    """
    An abstract base class for document parsers.
    
    This class defines the interface that all parser implementations must follow,
    ensuring that they can be used interchangeably within the ingestion pipeline.
    """

    @abc.abstractmethod
    def parse(self, file_path: str, file_metadata: Dict[str, Any]) -> List[TextNode]:
        """
        Parses a file and converts it into a list of TextNode objects.

        Args:
            file_path (str): The path to the file to be parsed.
            file_metadata (Dict[str, Any]): Metadata about the file.

        Returns:
            List[TextNode]: A list of TextNode objects, each representing a chunk of the document.
        """
        pass