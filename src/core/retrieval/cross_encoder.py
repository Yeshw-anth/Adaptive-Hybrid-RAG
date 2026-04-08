from typing import List
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks search results using a CrossEncoder model, preserving the full
    NodeWithScore object structure.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the Reranker.
        """
        logger.info(f"Initializing CrossEncoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("CrossEncoder model loaded successfully.")

    def rerank_nodes(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Reranks a list of NodeWithScore objects based on a query.

        Args:
            query (str): The search query.
            nodes (List[NodeWithScore]): A list of nodes to rerank.

        Returns:
            List[NodeWithScore]: The reranked list of nodes, with updated scores.
        """
        if not nodes:
            return []

        # Create pairs of [query, node_text] for scoring
        pairs = [[query, node.get_text()] for node in nodes]
        
        # Predict scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Assign new scores back to the nodes
        for node, score in zip(nodes, scores):
            node.score = float(score) # Update the node's score in-place
        
        # Sort the original list of nodes by their new scores
        nodes.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Reranked {len(nodes)} nodes successfully.")
        return nodes