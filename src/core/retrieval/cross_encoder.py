from typing import List

from llama_index.core.schema import NodeWithScore
from src.core.logging_config import logger
from sentence_transformers import CrossEncoder



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
        Reranks a list of nodes based on their relevance to the query using the cross-encoder.

        Args:
            query (str): The search query.
            nodes (List[NodeWithScore]): A list of nodes to rerank.

        Returns:
            List[NodeWithScore]: The reranked list of nodes, with updated scores.
        """
        if not nodes:
            return []

        logger.info(f"Reranking {len(nodes)} nodes for query: '{query[:50]}...'")

        # Create pairs of [query, node_text] for scoring
        pairs = [[query, node.get_text()] for node in nodes]

        # Score the pairs
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}", exc_info=True)
            # Return original nodes if reranking fails
            return nodes

        # Update node scores with the new cross-encoder scores
        for node, score in zip(nodes, scores):
            node.score = float(score)

        # Sort nodes by the new score in descending order
        reranked_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)

        logger.info("Successfully reranked nodes.")
        return reranked_nodes