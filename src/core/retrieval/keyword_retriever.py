import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode, NodeWithScore

logger = logging.getLogger(__name__)

class KeywordRetriever:
    """
    A retriever based on the BM25 algorithm for efficient keyword-based search.
    """
    def __init__(self, nodes: List[TextNode]):
        self.nodes = []
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.set_nodes(nodes)

    def set_nodes(self, nodes: List[TextNode]):
        """Updates the documents in the retriever and re-initializes the BM25 index."""
        self.nodes = nodes
        if not nodes:
            logger.warning("KeywordRetriever has been set with zero nodes. It will not be functional until nodes are provided.")
            self.corpus = []
            self.tokenized_corpus = []
            self.bm25 = None
        else:
            self.corpus = [node.get_content() for node in nodes]
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"KeywordRetriever re-initialized with {len(nodes)} nodes.")

    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieves the top_k most relevant documents for a given query
        using BM25.

        Args:
            query (str): The search query.
            top_k (int): The number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of retrieved documents, formatted as dictionaries.
        """
        if self.bm25 is None:
            logger.warning("KeywordRetriever has no documents indexed. Returning empty list.")
            return []
            
        logger.info(f"Performing keyword retrieval for query: '{query}'")
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get the top_k indices and scores
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        retrieved_nodes = []
        for i in top_indices:
            node = self.nodes[i]
            retrieved_nodes.append(NodeWithScore(node=node, score=doc_scores[i]))
        
        logger.info(f"Retrieved {len(retrieved_nodes)} documents using keyword search.")
        return retrieved_nodes