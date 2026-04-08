import logging
import time
from typing import Dict, Any

from src.core.pipelines.base import Pipeline
from src.core.retrieval.keyword_retriever import KeywordRetriever
from src.core.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class KeywordPipeline(Pipeline):
    """
    A pipeline that uses keyword-based retrieval (BM25) to find documents
    and a generator LLM to synthesize an answer. It's designed for speed
    and precision on keyword-heavy queries.
    """
    def __init__(self, keyword_retriever: KeywordRetriever, llm_client: OllamaClient):
        self.retriever = keyword_retriever
        self.llm_client = llm_client
        self.prompt_template = """Here is the user's query: "{query}"
        
Here are the most relevant documents found based on keyword search:
{context}

Based on these documents, please provide a direct and concise answer to the user's query.
Answer:"""

    async def execute(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the keyword-based RAG pipeline.
        """
        start_time = time.time()
        logger.info(f"Executing KeywordPipeline for query: '{query}'")

        # 1. Retrieve documents using keyword search
        retrieved_docs = self.retriever.retrieve(query, top_k=5)
        
        if not retrieved_docs:
            logger.warning("KeywordPipeline: No documents found for the query. Returning empty response.")
            return {
                "query": query,
                "answer": "I could not find any information related to your query.",
                "context": "",
                "retrieved_docs": [],
                "latency": time.time() - start_time,
                "pipeline": "keyword",
                "query_metadata": query_analysis.get("metadata", {}),
                "strategy": query_analysis.get("strategy", {}),
            }

        # Format context for the LLM
        context_str = "\n\n".join([f"Document (ID: {doc['id']}):\n{doc['text']}" for doc in retrieved_docs])

        # 2. Generate a response using the LLM
        prompt = self.prompt_template.format(query=query, context=context_str)
        answer = await self.llm_client.generate_from_prompt(prompt)

        end_time = time.time()
        latency = end_time - start_time

        result = {
            "query": query,
            "answer": answer,
            "context": context_str,
            "retrieved_docs": retrieved_docs,
            "latency": latency,
            "pipeline": "keyword",
            "query_metadata": query_analysis.get("metadata", {}),
            "strategy": query_analysis.get("strategy", {}),
        }
        
        logger.info(f"KeywordPipeline executed in {latency:.2f} seconds.")
        return result