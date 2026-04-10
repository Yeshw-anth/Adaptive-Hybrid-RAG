import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.core.retrieval.retriever import Retriever
from llama_index.core.schema import NodeWithScore, TextNode
from typing import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval to improve search results.
    This approach leverages both semantic meaning and keyword relevance.
    """
    def __init__(self, vector_retriever: Retriever, all_docs: List[TextNode]):
        self.vector_retriever = vector_retriever
        self.all_nodes: List[TextNode] = []
        self.bm25 = None
        self.update_corpus(all_docs)

    def update_corpus(self, all_nodes: List[TextNode]):
        """
        Initializes or updates the BM25 model with a new corpus of TextNode objects.
        """
        self.all_nodes = all_nodes
        self.nodes_by_id = {node.id_: node for node in all_nodes}
        
        doc_contents = [node.get_content() for node in all_nodes]
        
        if not doc_contents or not any(doc_contents):
            self.bm25 = None
            logger.warning("BM25 model not initialized or updated because the corpus is empty.")
            return

        tokenized_corpus = [doc.split(" ") for doc in doc_contents]
        
        if not any(tokenized_corpus):
            self.bm25 = None
            logger.warning("BM25 model not initialized: Corpus contains only empty documents.")
            return

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 model initialized/updated with {len(doc_contents)} documents.")

    def retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5, filters: Dict[str, Any] = None) -> List[NodeWithScore]:
        """
        Performs hybrid retrieval by combining vector and BM25 scores.
        Returns a list of NodeWithScore objects.
        """
        logger.info(f"Performing hybrid retrieval for query: '{query}' with top_k={top_k}, alpha={alpha}, filters={filters}")

        # 1. Dense retrieval (Vector Search)
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2, filters=filters)
        vector_scores = {res.node.id_: res.score for res in vector_results}
        
        if not vector_results:
            logger.warning("Vector search returned no results. Cannot perform hybrid search.")
            return []

        # 2. Sparse retrieval (BM25)
        if not self.bm25:
            logger.warning("BM25 model is not available. Falling back to pure vector search.")
            return sorted(vector_results, key=lambda x: x.score, reverse=True)[:top_k]

        tokenized_query = query.split(" ")
        bm25_scores_full = self.bm25.get_scores(tokenized_query)

        bm25_scores = {node.id_: score for node, score in zip(self.all_nodes, bm25_scores_full)}

        # 3. Combine scores using Reciprocal Rank Fusion (RRF)
        # RRF uses the rank of the documents, not their scores.
        
        # Create ranked lists of node IDs
        vector_ranked_ids = [res.node.id_ for res in sorted(vector_results, key=lambda x: x.score, reverse=True)]
        
        # Create a dictionary of BM25 scores for all nodes
        bm25_scores_all = {node.id_: score for node, score in zip(self.all_nodes, bm25_scores_full)}
        bm25_ranked_ids = [
            node_id for node_id, score in sorted(bm25_scores_all.items(), key=lambda item: item[1], reverse=True)
        ]

        # Calculate RRF scores
        rrf_scores = {}
        all_node_ids = set(vector_ranked_ids) | set(bm25_ranked_ids)
        
        k = 60  # Constant for RRF, as recommended in the original paper

        for node_id in all_node_ids:
            rrf_score = 0.0
            if node_id in vector_ranked_ids:
                rank = vector_ranked_ids.index(node_id) + 1
                rrf_score += 1 / (k + rank)
            
            if node_id in bm25_ranked_ids:
                rank = bm25_ranked_ids.index(node_id) + 1
                rrf_score += 1 / (k + rank)
            
            rrf_scores[node_id] = rrf_score

        # Sort by the combined RRF score (higher is better)
        sorted_node_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        # 4. Create NodeWithScore objects for the top results
        top_nodes = []
        for node_id in sorted_node_ids:
            # Find the original NodeWithScore from the vector results to preserve its score and metadata
            # If not in vector results, create a new one from the master node list
            original_node = next((res for res in vector_results if res.node.id_ == node_id), None)
            if original_node:
                # We use the RRF score for ranking, but can keep the original vector score for context
                node_with_new_score = NodeWithScore(node=original_node.node, score=rrf_scores.get(node_id))
                top_nodes.append(node_with_new_score)
            elif node_id in self.nodes_by_id:
                # This case handles nodes found by BM25 but not by vector search
                node = self.nodes_by_id[node_id]
                score = rrf_scores.get(node_id)
                top_nodes.append(NodeWithScore(node=node, score=score))

        logger.info(f"Hybrid retrieval found {len(top_nodes)} documents.")
        return top_nodes