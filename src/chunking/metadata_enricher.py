import uuid
from typing import List
from llama_index.core.schema import TextNode

class MetadataEnricher:
    """
    A component responsible for enriching TextNodes with valuable metadata.
    """
    def enrich_nodes(self, nodes: List[TextNode], file_name: str, file_path: str, section_title: str) -> List[TextNode]:
        """
        Adds standardized metadata to a list of nodes.
        """
        for node in nodes:
            node.metadata["file_name"] = file_name
            node.metadata["file_path"] = file_path
            node.metadata["section_title"] = section_title
            node.metadata["chunk_id"] = str(uuid.uuid4())
        return nodes