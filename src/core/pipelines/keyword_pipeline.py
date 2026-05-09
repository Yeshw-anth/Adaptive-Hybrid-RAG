from src.core.logging_config import logger
import time
from typing import Dict, Any

from src.core.pipelines.base import Pipeline
from src.core.retrieval.keyword_retriever import KeywordRetriever
from src.core.llm.ollama_client import OllamaClient

class KeywordPipeline(Pipeline):
    """
    A pipeline that uses keyword-based retrieval (BM25) to find documents
    and a generator LLM to synthesize an answer. It's designed for speed
    and precision on keyword-heavy queries.
    """
    def __init__(self, keyword_retriever: KeywordRetriever, llm_client: OllamaClient):
        self.retriever = keyword_retriever
        self.llm_client = llm_client

    async def execute(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the keyword-based RAG pipeline.
        """
        start_time = time.time()
        strategy = query_analysis["strategy"]
        logger.info(f"Executing KeywordPipeline for query: '{query}' with strategy: {strategy.name}")

        # 1. Retrieval
        retrieved_nodes = self.retriever.retrieve(query, top_k=strategy.top_k)
        if not retrieved_nodes:
            logger.warning("No nodes retrieved for the query.")
            return self._generate_empty_response(query, start_time, query_analysis)

        # 2. Synthesis
        response_text = await self._synthesize_response(query, retrieved_nodes, strategy)

        # 3. Post-processing and Formatting
        final_response = self._format_response(
            query=query,
            response_text=response_text,
            retrieved_nodes=retrieved_nodes,
            strategy=strategy,
            start_time=start_time,
            query_analysis=query_analysis
        )
        
        logger.info(f"KeywordPipeline execution finished in {final_response['latency']:.2f} seconds.")
        return final_response

    async def _synthesize_response(self, query: str, nodes: list, strategy: any) -> str:
        """Synthesizes a response from the retrieved nodes."""
        context = "\n\n".join([node['text'] for node in nodes])
        
        logger.debug(f"Synthesizing response with model: {strategy.model}")
        return await self.llm_client.generate_structured_response(context=context, query=query, model=strategy.model)

    def _format_response(self, **kwargs) -> Dict[str, Any]:
        """Formats the final response dictionary."""
        latency = time.time() - kwargs['start_time']
        
        response = {
            "query": kwargs['query'],
            "answer": kwargs['response_text'],
            "context": "\n\n".join([node['text'] for node in kwargs['retrieved_nodes']]),
            "retrieved_docs": kwargs['retrieved_nodes'],
            "latency": latency,
            "pipeline": "keyword",
            "query_metadata": kwargs['query_analysis'].get("metadata", {}),
            "strategy": kwargs['strategy'],
        }
        return response

    def _generate_empty_response(self, query: str, start_time: float, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generates an empty response when no nodes are retrieved."""
        return {
            "query": query,
            "answer": "Could not find relevant information using keywords.",
            "context": "",
            "retrieved_docs": [],
            "latency": time.time() - start_time,
            "pipeline": "keyword",
            "query_metadata": query_analysis.get("metadata", {}),
            "strategy": query_analysis.get("strategy", {}),
        }