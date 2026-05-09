from src.core.logging_config import logger
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode, NodeWithScore

class KeywordRetriever:
    """
    A retriever based on the BM25 algorithm for efficient keyword-based search.
    """
    def __init__(self, all_nodes: List[TextNode]):
        self.all_nodes = all_nodes
        self.bm25 = None
        if all_nodes:
            self.update_corpus([node.get_content() for node in all_nodes])
        logger.info("KeywordRetriever initialized.")

    def update_corpus(self, corpus: List[str]):
        """Updates the BM25 model with a new corpus."""
        if not corpus or not any(corpus):
            self.bm25 = None
            logger.warning("BM25 model not initialized or updated because the corpus is empty.")
            return
        
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 model updated with a corpus of {len(corpus)} documents.")

    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieves the top_k most relevant documents for a given query
        using BM25.
        """
        if self.bm25 is None:
            logger.warning("BM25 model is not available. Cannot perform retrieval.")
            return []

        logger.info(f"Performing keyword retrieval for query: '{query}'")
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_n_indices:
            # Ensure the score is a float, as BM25 can return numpy types
            score = float(doc_scores[i])
            if score > 0: # Only return documents with a positive score
                node = self.all_nodes[i]
                results.append(NodeWithScore(node=node, score=score))
            
        logger.info(f"Retrieved {len(results)} documents with positive scores for query.")
        return results