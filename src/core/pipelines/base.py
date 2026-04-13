from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from llama_index.core.schema import NodeWithScore
from src.core.caching.response_cache import ResponseCache
import logging

class Pipeline(ABC):
    """Abstract base class for all RAG pipelines."""

    @abstractmethod
    async def execute(self, query: str, query_analysis: Dict[str, Any], cache: Optional[ResponseCache] = None) -> Dict[str, Any]:
        """
        Executes the pipeline for a given query.

        Args:
            query (str): The user's query.
            query_analysis (Dict[str, Any]): The analysis of the query from the QueryAnalyzer.
            cache (Optional[ResponseCache]): The response cache to use for this query.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the pipeline execution.
        """
        pass

    def _format_nodes_to_docs(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """
        Converts a list of NodeWithScore objects into a list of dictionaries
        formatted for logging and API responses.
        This function is robust and can handle lists that are already formatted.
        """
        if not nodes:
            return []

        # If the list already consists of dictionaries, return it directly.
        if isinstance(nodes[0], dict):
            return nodes

        # Otherwise, assume it's a list of NodeWithScore objects and convert them.
        formatted_docs = []
        for node in nodes:
            if hasattr(node, 'node') and hasattr(node, 'score'):
                formatted_docs.append({
                    "text": node.node.get_text(),
                    "score": node.score,
                    "metadata": node.node.metadata or {}
                })
            else:
                # This case should ideally not be hit if the list is homogeneous
                logging.warning(f"Unexpected type in _format_nodes_to_docs: {type(node)}")
        return formatted_docs