
import time
import logging
import json
from typing import Dict, Any, List

from src.core.pipelines.base import Pipeline
from src.core.retrieval.retriever import Retriever
from src.core.llm.ollama_client import OllamaClient
from src.core.decision.confidence_engine import ConfidenceEngine
from src.config import settings

class FastPipeline(Pipeline):
    """
    A fast RAG pipeline that prioritizes speed over depth.
    It uses basic vector retrieval and no reranking or expansion.
    """
    def __init__(self, retriever: Retriever, llm_client: OllamaClient, confidence_engine: ConfidenceEngine):
        self.retriever = retriever
        self.llm_client = llm_client
        self.confidence_engine = confidence_engine
        self.prompt_template = (
            "You are a helpful assistant. Your task is to answer the user's query based *only* on the provided context. "
            "Do not use any outside knowledge. If the context does not contain the answer, state that you cannot answer based on the information given.\n\n"
            "--- CONTEXT ---\n"
            "{context}\n"
            "--- END CONTEXT ---\n\n"
            "User Query: {query}\n"
            "Answer:"
        )

    async def execute(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        logging.info(f"Running FastPipeline for query: '{query}'")

        # 1. Retrieve documents
        retrieval_start = time.time()
        retrieved_nodes = self.retriever.retrieve(
            query,
            top_k=query_analysis['strategy'].get('top_k', settings.FAST_PIPELINE_TOP_K)
        )
        retrieval_time = time.time() - retrieval_start

        # The retriever returns a list of dicts. Access keys directly.
        retrieved_docs = [
            {
                "text": node.get('text', ''),
                "metadata": node.get('metadata', {}),
                "score": node.get('score', 0.0)
            }
            for node in retrieved_nodes
        ]

        serializable_docs = [
            {"text": doc.get('text'), "metadata": doc.get('metadata'), "score": doc.get('score')}
            for doc in retrieved_docs
        ]
        logging.info(f"RETRIEVED_CHUNKS: {json.dumps(serializable_docs, indent=2)}")

        if not retrieved_docs:
            return self._format_empty_response(query, start_time, query_analysis['metadata'])

        # 2. Construct context
        context = self._construct_context(retrieved_docs)

        # 3. Calculate context confidence and decide action
        confidence_result = self.confidence_engine.calculate_context_confidence(
            query_metadata=query_analysis['metadata'],
            retrieved_docs=retrieved_docs
        )
        action = self.confidence_engine.decide_action_from_context(confidence_result)

        if action == "abstain":
            logging.warning("Abstaining from answering due to low context confidence.")
            return self._format_empty_response(query, start_time, query_analysis['metadata'])

        # 4. Generate full response
        generation_start = time.time()
        try:
            # Use the new, stricter prompt template
            prompt = self.prompt_template.format(context=context, query=query)
            answer = await self.llm_client.generate_from_prompt(
                prompt,
                model=query_analysis['strategy'].get('model', settings.DEFAULT_LLM_MODEL)
            )
        except Exception as e:
            logging.error(f"Error during LLM generation in FastPipeline: {e}")
            return self._format_error_response(str(e), start_time, query_analysis['metadata'])
        generation_time = time.time() - generation_start

        # 6. Format and return response
        response = {
            "answer": answer,
            "sources": self._format_sources(retrieved_docs),
            "retrieved_docs": retrieved_docs,
            "latency": time.time() - start_time,
            "query_metadata": query_analysis['metadata'],
            "strategy": query_analysis['strategy'],
            "pipeline": "fast",
            "confidence_score": confidence_result.get("context_score", 0.0),
            "final_prompt": prompt
        }
        logging.info(f"GENERATED_RESPONSE: {json.dumps(response, indent=2, default=str)}")
        return response

    def _construct_context(self, docs: List[Dict]) -> str:
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.get('text', '')
            part = f"Context {i+1}:\n{content}"
            context_parts.append(part)
        return "\n\n".join(context_parts)

    def _format_sources(self, docs: List[Dict]) -> List[Dict]:
        sources = []
        for doc in docs:
            metadata = doc.get('metadata', {})
            sources.append({
                "file": metadata.get("file_name", "Unknown"),
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