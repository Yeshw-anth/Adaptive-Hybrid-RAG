import logging
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

class Retriever:
    """
    A class to handle the retrieval pipeline using a LlamaIndex VectorStoreIndex.
    """

    def __init__(self, vector_index: VectorStoreIndex):
        """
        Initializes the Retriever.

        Args:
            vector_index (VectorStoreIndex): An instance of a LlamaIndex VectorStoreIndex.
        """
        self.vector_index = vector_index

    def retrieve(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks for a given query, with optional metadata filters.

        Args:
            query (str): The user's query.
            top_k (int): The number of chunks to retrieve.
            filters (Dict[str, Any], optional): Metadata filters to apply.
                                                Example: {"section_title": "Skills"}

        Returns:
            List[Dict[str, Any]]: A list of retrieved chunk documents with their scores.
        """
        if not query:
            return []

        # Determine retrieval size: fetch more if we need to filter in-memory
        retrieval_k = top_k * 5 if filters else top_k

        index_retriever = self.vector_index.as_retriever(similarity_top_k=retrieval_k)

        # Retrieve nodes (without filters at the DB level)
        retrieved_nodes: List[NodeWithScore] = index_retriever.retrieve(query)
        logging.info(f"Retrieved {len(retrieved_nodes)} initial nodes from Faiss.")

        # In-memory filtering if filters are provided
        if filters:
            logging.info(f"Applying in-memory metadata filters: {filters}")
            filtered_nodes = []
            for node in retrieved_nodes:
                matches = all(node.node.metadata.get(key) == value for key, value in filters.items())
                if matches:
                    filtered_nodes.append(node)

            retrieved_nodes = filtered_nodes
            logging.info(f"Found {len(retrieved_nodes)} nodes after in-memory filtering.")

        # Trim to the desired top_k after filtering
        final_nodes = retrieved_nodes[:top_k]

        # Format the results
        search_results = [
            {
                "text": node.node.get_content(),
                "metadata": node.node.metadata,
                "score": node.score
            }
            for node in final_nodes
        ]
            
        return search_results