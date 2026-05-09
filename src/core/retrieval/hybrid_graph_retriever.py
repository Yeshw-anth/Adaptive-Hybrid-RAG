from src.core.logging_config import logger
import asyncio
from typing import List, Dict, Any

from src.core.retrieval.retriever import Retriever
from src.core.retrieval.graph_retriever import GraphRetriever
from src.data.schemas import Document



class HybridGraphRetriever:
    """
    A retriever that combines results from both vector search and graph traversal.
    """
    def __init__(self, vector_retriever: Retriever, graph_retriever: GraphRetriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Asynchronously retrieves documents from both vector and graph stores, then combines them.
        """
        logger.info(f"Executing hybrid graph retrieval for query: '{query}'")
        
        # 1. Asynchronously retrieve from both vector and graph retrievers
        # Note: We assume aretrieve methods exist on the retriever instances.
        vector_results, graph_results = await asyncio.gather(
            self.vector_retriever.aretrieve(query, top_k=top_k),
            self.graph_retriever.aretrieve(query, top_k=top_k)
        )
        
        logger.info(f"Retrieved {len(vector_results)} documents from vector store.")
        logger.info(f"Retrieved {len(graph_results)} documents from graph store.")

        # 2. Combine and deduplicate results
        combined_results = self._combine_and_deduplicate(vector_results, graph_results)
        
        logger.info(f"Combined and deduplicated to {len(combined_results)} documents.")
        
        # 3. Optionally, rerank the combined results (not implemented in this example)
        # For now, we just return the combined list.
        
        return combined_results

    def _combine_and_deduplicate(self, list1: List[Document], list2: List[Document]) -> List[Document]:
        """Combines two lists of Documents and removes duplicates based on document ID."""
        seen_ids = set()
        combined = []
        
        for doc in list1 + list2:
            # Assuming Document schema has an 'id' attribute for deduplication
            if hasattr(doc, 'id') and doc.id not in seen_ids:
                seen_ids.add(doc.id)
                combined.append(doc)
            elif doc.text not in seen_ids: # Fallback to text if no id
                seen_ids.add(doc.text)
                combined.append(doc)
                
        return combined