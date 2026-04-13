from typing import Any, Dict, List
import logging
from src.data.schemas import Document
from src.core.pipelines.base import Pipeline
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.cross_encoder import CrossEncoderReranker
from llama_index.core.schema import NodeWithScore, TextNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodePipeline(Pipeline):
    """
    A RAG pipeline optimized for code-related queries.
    This pipeline uses a hybrid retrieval approach and a reranker to find the most
    relevant code snippets.
    """

    def __init__(self, retriever: HybridRetriever, reranker: CrossEncoderReranker):
        self.retriever = retriever
        self.reranker = reranker

    async def execute(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the code-optimized RAG pipeline.
        Args:
            query: The user's query.
            query_analysis: A dictionary containing the query metadata and strategy.
        Returns:
            A dictionary with 'retrieved_docs' and 'reranked_docs'.
        """
        logger.info(f"--- Running CodePipeline for query: '{query}' ---")
        
        strategy: "Strategy" = query_analysis['strategy']

        # 1. Retrieval
        top_k = strategy.top_k
        logger.info(f"Step 1: Retrieving documents with top_k={top_k}.")
        retrieved_nodes = self.retriever.retrieve(query, top_k=top_k)
        retrieved_docs = self._format_nodes_to_docs(retrieved_nodes)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")

        # 2. Reranking
        if strategy.use_reranker and retrieved_docs: # Reranking is default for code
            logger.info("Step 2: Reranking documents.")
            reranked_nodes = self.reranker.rerank_nodes(query, retrieved_nodes)
            reranked_docs = self._format_nodes_to_docs(reranked_nodes)
            logger.info(f"Reranked down to {len(reranked_docs)} documents.")
        else:
            reranked_docs = retrieved_docs
            logger.info("Step 2: Skipping reranker based on strategy.")

        return {
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs
        }

    def _format_nodes_to_docs(self, nodes: List[NodeWithScore]) -> List[Document]:
        """Converts LlamaIndex nodes to our internal Document format."""
        docs = []
        for node in nodes:
            # Ensure node.node is a TextNode and has metadata
            if isinstance(node.node, TextNode) and hasattr(node.node, 'metadata'):
                doc = Document(
                    text=node.node.get_content(),
                    metadata=node.node.metadata,
                    score=node.score
                )
                docs.append(doc)
            else:
                logger.warning(f"Skipping node of type {type(node.node)} without expected structure.")
        return docs