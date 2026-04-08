import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.core.retrieval.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval to improve search results.
    This approach leverages both semantic meaning and keyword relevance.
    """
    def __init__(self, vector_retriever: Retriever, all_docs: List[Dict[str, Any]]):
        self.vector_retriever = vector_retriever
        self.all_docs = []
        self.doc_contents = []
        self.bm25 = None
        self.set_nodes(all_docs)

    def set_nodes(self, nodes: List[Any]):
        """
        Updates the corpus for the BM25 part of the retriever.
        Accepts LlamaIndex TextNode objects or dictionaries.
        """
        if not nodes:
            self.update_corpus([])
            return

        # Convert TextNode objects to the dictionary format this retriever expects
        if hasattr(nodes[0], 'to_dict'): # Check if it's a LlamaIndex object
            all_docs_as_dicts = [node.to_dict()['node'] for node in nodes]
            # The 'content' is nested inside the 'text' key for nodes
            for doc_dict in all_docs_as_dicts:
                doc_dict['content'] = doc_dict.get('text', '')
        else:
            all_docs_as_dicts = nodes # Assume it's already a list of dicts

        self.update_corpus(all_docs_as_dicts)

    def update_corpus(self, all_docs: List[Dict[str, Any]]):
        """
        Initializes or updates the BM25 model with a new corpus.
        """
        self.all_docs = all_docs
        self.doc_contents = [doc.get('content', '') for doc in all_docs]
        
        if not self.doc_contents or not any(self.doc_contents):
            self.bm25 = None
            logger.warning("BM25 model not initialized or updated because the corpus is empty.")
            return

        tokenized_corpus = [doc.split(" ") for doc in self.doc_contents]
        
        # Filter out empty documents that can cause errors in BM25
        if not any(tokenized_corpus):
            self.bm25 = None
            logger.warning("BM25 model not initialized: Corpus contains only empty documents.")
            return

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 model initialized/updated with {len(self.doc_contents)} documents.")

    def retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval by combining vector and BM25 scores.

        Args:
            query: The user's query.
            top_k: The number of documents to return.
            alpha: The weighting factor for combining scores (0=BM25, 1=vector).
            filters: Metadata filters to apply to the retrieval.

        Returns:
            A list of the top-k documents after hybrid scoring.
        """
        logger.info(f"Performing hybrid retrieval for query: '{query}' with top_k={top_k}, alpha={alpha}, filters={filters}")

        # 1. Dense retrieval (Vector Search) with filters
        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2, filters=filters)
        vector_scores = {res['metadata']['chunk_id']: res['score'] for res in vector_results}
        
        # If vector search with filters returns nothing, we can't proceed.
        if not vector_results:
            logger.warning("Hybrid retrieval: Vector search with filters returned no results.")
            return []

        # The corpus for BM25 should be the documents that matched the vector search (and thus the filter)
        # This is more efficient and correct than searching the whole corpus.
        candidate_docs = [res for res in self.all_docs if res['metadata']['chunk_id'] in vector_scores]
        if not candidate_docs:
            logger.warning("Hybrid retrieval: No candidate documents found after vector search.")
            return []
            
        candidate_contents = [doc.get('content', '') for doc in candidate_docs]
        tokenized_corpus = [doc.split(" ") for doc in candidate_contents]

        # Create a temporary BM25 index on the filtered candidate documents
        if not any(tokenized_corpus):
            logger.warning("BM25 model not initialized for this query: Candidate corpus is empty.")
            # Fallback to just returning the vector results
            return sorted(vector_results, key=lambda x: x['score'], reverse=True)[:top_k]

        bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Sparse retrieval (BM25) on the filtered set
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        logger.info(f"Vector search found {len(vector_results)} results.")
        logger.info(f"BM25 search will be performed on {len(candidate_docs)} candidate documents.")

        # Combine scores
        combined_scores = {}
        for i, doc in enumerate(candidate_docs):
            chunk_id = doc['metadata']['chunk_id']
            
            # Normalize scores (simple min-max, can be improved)
            vec_score = vector_scores.get(chunk_id, 0)
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0
            
            # Simple weighted combination
            combined_scores[chunk_id] = (alpha * vec_score) + ((1 - alpha) * bm25_score)

        # Sort and get top-k results
        sorted_chunk_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
        
        # Map chunk_ids back to documents
        results_map = {doc['metadata']['chunk_id']: doc for doc in candidate_docs}
        top_docs = [results_map[chunk_id] for chunk_id in sorted_chunk_ids if chunk_id in results_map]
        
        logger.info(f"Hybrid retrieval found {len(top_docs)} documents.")
        return top_docs