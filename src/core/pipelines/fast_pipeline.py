
import time
from src.core.logging_config import logger
import json
from typing import Dict, Any, List, Optional

from src.core.pipelines.base import Pipeline
from src.core.retrieval.retriever import Retriever
from src.core.retrieval.graph_retriever import GraphRetriever
from src.core.llm.ollama_client import OllamaClient
from src.core.decision.confidence_engine import ConfidenceEngine
from src.core.caching.response_cache import ResponseCache
from src.config import settings
from llama_index.core.schema import NodeWithScore, TextNode

class FastPipeline(Pipeline):
    """
    A fast RAG pipeline that prioritizes speed over depth.
    It uses basic vector retrieval and no reranking or expansion.
    """
    def __init__(self, retriever: Retriever, graph_retriever: GraphRetriever, llm_client: OllamaClient, confidence_engine: ConfidenceEngine):
        self.retriever = retriever
        self.graph_retriever = graph_retriever
        self.llm_client = llm_client
        self.confidence_engine = confidence_engine
        self.prompt_template = (
            "**System Prompt: You are a senior AI systems engineer.**\n\n"
            "Your task is to provide a clear, concise, and structured answer to the user's query, "
            "basing your response *exclusively* on the context provided below. Do not use any external knowledge.\n\n"
            "**Instructions:**\n"
            "1.  **Analyze the Context:** Carefully review all the provided context documents.\n"
            "2.  **Synthesize the Answer:** Formulate a direct answer to the user's query based on the information in the context.\n"
            "3.  **Cite Sources:** For each piece of information you use, you MUST cite the corresponding context document using the format `[Source X]`, where 'X' is the context number.\n"
            "4.  **Structure the Output:** Format your response into two sections:\n"
            "    *   **Direct Answer:** A concise, immediate answer to the user's question.\n"
            "    *   **Detailed Explanation:** A more thorough explanation, elaborating on the answer and synthesizing information from multiple sources. Ensure all claims are supported by citations.\n"
            "5.  **Handle Missing Information:** If the context does not contain the information needed to answer the query, state clearly: 'The provided context does not contain enough information to answer this question.' Do not attempt to answer.\n\n"
            "--- CONTEXT ---\n"
            "{context}\n"
            "--- END CONTEXT ---\n\n"
            "**User Query:** {query}\n\n"
            "**Your Response:**"
        )

    async def execute(self, query: str, query_analysis: Dict[str, Any], cache: Optional[ResponseCache] = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Running FastPipeline for query: '{query}'")

        # 1. Determine retrieval strategy
        retrieval_strategy = query_analysis['strategy'].retrieval_strategy
        logger.info(f"FastPipeline using retrieval strategy: {retrieval_strategy}")

        # 2. Retrieve documents
        retrieval_start = time.time()
        context = ""
        retrieved_nodes = []

        # A hybrid strategy involves using the graph retriever first.
        if retrieval_strategy == 'hybrid':
            logger.info("Hybrid strategy detected. Attempting graph retrieval first.")
            graph_context = await self.graph_retriever.retrieve(query)
            if graph_context:
                context = graph_context
                # Create a dummy node for citation purposes
                retrieved_nodes_with_score = [
                    NodeWithScore(node=TextNode(text=graph_context, id_="graph_context", metadata={"file_path": "Knowledge Graph"}), score=1.0)
                ]
                retrieved_nodes = self._format_nodes_to_docs(retrieved_nodes_with_score)
            else:
                logger.info("Graph retrieval yielded no results. Falling back to vector retrieval.")
                retrieved_nodes_with_score = self.retriever.retrieve(
                    query,
                    top_k=query_analysis['strategy'].top_k
                )
                if retrieved_nodes_with_score:
                    context = self._construct_context_from_nodes(retrieved_nodes_with_score)
                    retrieved_nodes = self._format_nodes_to_docs(retrieved_nodes_with_score)
        
        else: # Default to vector retrieval
            retrieved_nodes_with_score = self.retriever.retrieve(
                query,
                top_k=query_analysis['strategy'].top_k
            )
            if retrieved_nodes_with_score:
                context = self._construct_context_from_nodes(retrieved_nodes_with_score)
                retrieved_nodes = self._format_nodes_to_docs(retrieved_nodes_with_score)

        retrieval_time = time.time() - retrieval_start
        
        if not context:
            return self._format_empty_response(query, start_time, query_analysis)

        # Caching logic
        if cache:
            node_ids = sorted([node['id'] for node in retrieved_nodes])
            cached_response = cache.get(query, node_ids)
            if cached_response:
                logger.info(f"Cache hit for query: '{query}'. Returning cached response.")
                cached_response["latency"] = time.time() - start_time
                cached_response["pipeline"] = "fast-cached"
                return cached_response
            logger.info(f"Cache miss for query: '{query}'. Proceeding with generation.")

        serializable_docs = retrieved_nodes
        logger.info(f"RETRIEVED_CHUNKS: {json.dumps(serializable_docs, indent=2)}")

        # 3. Calculate context confidence and decide action
        # Note: Confidence calculation might need adjustment for graph context
        confidence_result = self.confidence_engine.calculate_context_confidence(
            query_metadata=query_analysis['metadata'],
            retrieved_docs=retrieved_nodes # This needs to be adaptable
        )
        action = self.confidence_engine.decide_action_from_context(confidence_result)

        if action == "abstain":
            logger.warning("Abstaining from answering due to low context confidence.")
            return self._format_empty_response(query, start_time, query_analysis)

        # 4. Generate full response
        generation_start = time.time()
        try:
            answer = await self.llm_client.generate_structured_response(
                context,
                query,
                model=query_analysis['strategy'].model
            )
        except Exception as e:
            logger.error(f"Error during LLM generation in FastPipeline: {e}")
            return self._format_error_response(str(e), start_time, query_analysis)
        generation_time = time.time() - generation_start

        # 5. Format and return response
        final_retrieved_docs = retrieved_nodes
        response = {
            "answer": answer,
            "sources": self._format_sources(final_retrieved_docs),
            "retrieved_docs": final_retrieved_docs,
            "latency": time.time() - start_time,
            "query_metadata": query_analysis['metadata'],
            "strategy": query_analysis['strategy'],
            "pipeline": "fast",
            "confidence_score": confidence_result.get("context_score", 0.0),
            "context": context
        }
        
        if cache:
            node_ids = sorted([doc['id'] for doc in final_retrieved_docs])
            cache.set(query, node_ids, response)
            logger.info(f"Response for query '{query}' stored in cache.")
            
        logger.info(f"GENERATED_RESPONSE: {json.dumps(response, indent=2, default=str)}")
        return response

    def _construct_context_from_nodes(self, nodes: List[Any]) -> str:
        context_parts = []
        for i, node in enumerate(nodes):
            content = node.get_text()
            part = f"Context {i+1}:\n{content}"
            context_parts.append(part)
        return "\n\n".join(context_parts)

    def _format_nodes_to_docs(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """Converts a list of NodeWithScore to a list of standardized dictionaries."""
        if not nodes:
            return []
        
        formatted_docs = []
        for node in nodes:
            doc = {
                "id": node.node.id_,
                "text": node.get_text(),
                "metadata": node.metadata or {},
                "score": node.score,
                "file_path": node.metadata.get("file_path", "Unknown")
            }
            formatted_docs.append(doc)
        return formatted_docs

    def _format_sources(self, docs: List[Dict]) -> List[Dict]:
        sources = []
        for doc in docs:
            metadata = doc.get('metadata', {})
            sources.append({
                "file": metadata.get("file_path", "Unknown"), # Corrected from file_name
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "vector_score": float(doc.get("score", 0.0)),
            })
        return sources

    def _format_empty_response(self, query: str, start_time: float, query_analysis: Dict) -> Dict[str, Any]:
        return {
            "answer": "Could not find any relevant information.",
            "sources": [],
            "retrieved_docs": [],
            "final_docs": [],
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_analysis.get('metadata', {}),
            "strategy": query_analysis.get('strategy', {}),
            "pipeline": "fast",
        }

    def _format_error_response(self, error_message: str, start_time: float, query_analysis: Dict) -> Dict[str, Any]:
        return {
            "answer": f"An error occurred: {error_message}",
            "sources": [],
            "retrieved_docs": [],
            "final_docs": [],
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_analysis.get('metadata', {}),
            "strategy": query_analysis.get('strategy', {}),
            "pipeline": "fast",
        }