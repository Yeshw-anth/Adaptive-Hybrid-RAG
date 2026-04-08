# Force reload
import time
import logging
import json
from typing import Dict, Any, List, Optional

from src.data.schemas import Document
from src.core.pipelines.base import Pipeline
from src.core.retrieval.retriever import Retriever
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.cross_encoder import CrossEncoderReranker
from llama_index.core.schema import NodeWithScore, TextNode
from src.core.llm.ollama_client import OllamaClient
from src.core.strategy.query_expander import QueryExpander
from src.core.decision.confidence_engine import ConfidenceEngine

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from src.data.schemas import Document

class AccuratePipeline(Pipeline):
    """
    A sophisticated RAG pipeline that uses hybrid retrieval, reranking, and a
    confidence-driven expansion loop to provide high-quality answers.
    """

    def __init__(self, retriever: Retriever, hybrid_retriever: HybridRetriever, reranker: CrossEncoderReranker,
                 llm_client: OllamaClient, query_expander: QueryExpander, confidence_engine: ConfidenceEngine):
        self.retriever = retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.query_expander = query_expander
        self.confidence_engine = confidence_engine


    async def execute(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        logging.info(f"--- Running AccuratePipeline for query: '{query}' ---")

        # Determine chunking strategy from query analysis or default to semantic
        chunking_strategy = query_analysis.get('strategy', {}).get('chunking_strategy', 'semantic')
        logging.info(f"Using chunking strategy: {chunking_strategy}")

        # 1. Analyze query for potential section filters
        logging.info("Step 1: Analyzing query for section filters...")

        filters = None

        # 2. Initial Retrieval with potential filters
        logging.info("Step 2: Performing initial retrieval...")
        retrieval_result = self._retrieve_docs(query, query_analysis['strategy'], filters=filters)

        # Fallback to broad search if filtered search yields no results
        if not retrieval_result["docs"] and filters:
            logging.warning("Filtered retrieval yielded no results. Falling back to broad retrieval.")
            retrieval_result = self._retrieve_docs(query, query_analysis['strategy'], filters=None)

        if not retrieval_result["docs"]:
            logging.warning("No documents found during initial retrieval.")
            return self._format_empty_response(query, start_time, query_analysis)
        logging.info(f"Step 2: Retrieved {len(retrieval_result['docs'])} documents.")

        # 3. Rerank the initial documents
        logging.info("Step 3: Reranking retrieved documents...")
        rerank_result = self._rerank_docs(query, retrieval_result["docs"], query_analysis['strategy'])
        if rerank_result["docs"]:
            logging.info(f"Step 3: Reranking complete. Top document score: {rerank_result['docs'][0]['score']}")
        else:
            logging.warning("Reranking returned no documents.")

        # 4. Decide action and potentially expand
        logging.info("Step 4: Making decision on action (expand, generate, or abstain)...")
        decision_result = await self._decide_and_expand(
            query, query_analysis, retrieval_result["docs"], rerank_result["docs"], initial_filters=filters
        )

        action = decision_result["action"]
        if action == "abstain":
            return self._format_abstain_response(decision_result, start_time, query_analysis)

        # 5. Generate the final answer using the final set of documents
        final_docs = decision_result["final_docs"]

        # 5a. Fetch parent documents if the strategy requires it
        if query_analysis['strategy'].get('use_parent_child', False):
            logging.info("Step 5a: Strategy requires parent-child retrieval. Fetching parent documents.")
            final_docs = self._fetch_and_add_parents(final_docs)

        context = self._construct_context(final_docs)
        logging.info(f"CONTEXT_SENT_TO_LLM: {context}")

        # 6. Generate full response
        try:
            answer = await self.llm_client.generate_response(context, query, query_analysis['strategy'].get('model', settings.DEFAULT_LLM_MODEL))
        except Exception as e:
            logging.error(f"Error during LLM generation in AccuratePipeline: {e}")
            return self._format_error_response(str(e), start_time, query_analysis)

        # 7. Format the final response
        response = self._format_response(
            answer=answer,
            sources=self._format_sources(final_docs),
            retrieved_docs=retrieval_result["docs"],
            reranked_docs=rerank_result["docs"],
            final_docs=final_docs,
            context=context,
            latency=time.time() - start_time,
            query_metadata=query_analysis['metadata'],
            strategy=query_analysis['strategy'],
            pipeline="accurate",
            confidence=decision_result["confidence"],
            action_taken=action,
            expansion_details=decision_result["expansion_details"],
            confidence_before_expansion=decision_result.get("confidence_before_expansion", 0.0)
        )
        logging.info(f"GENERATED_RESPONSE: {json.dumps(response, indent=2)}")
        return response

    def _fetch_and_add_parents(self, child_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetches parent documents for a list of child documents and merges them.
        Now works with lists of dictionaries.
        """
        parent_ids_to_fetch = set()
        for doc in child_docs:
            if doc.get("metadata") and doc["metadata"].get("parent_id"):
                parent_ids_to_fetch.add(doc["metadata"]["parent_id"])

        if not parent_ids_to_fetch:
            logging.info("No parent documents to fetch.")
            return child_docs

        logging.info(f"Fetching {len(parent_ids_to_fetch)} parent documents.")

        parent_nodes = []
        for parent_id in parent_ids_to_fetch:
            retrieved = self.retriever.retrieve(query="", top_k=1, filters={"chunk_id": parent_id})
            if retrieved:
                parent_nodes.extend(retrieved)

        # Convert parent nodes to the standardized dictionary format
        parent_docs_as_dicts = self._format_nodes_to_docs(parent_nodes)

        logging.info(f"Successfully fetched {len(parent_docs_as_dicts)} parent documents.")

        # Merge and deduplicate child and parent documents
        combined_docs = self._merge_and_deduplicate(child_docs + parent_docs_as_dicts)

        return combined_docs


    def _retrieve_docs(self, query: str, strategy: Dict[str, Any], filters: Dict[str, str] = None) -> Dict:
        retrieval_start = time.time()
        retrieval_strategy = strategy.get("retrieval_strategy", "vector")
        depth = strategy.get("top_k", settings.RETRIEVER_TOP_K)

        if retrieval_strategy == 'hybrid':
            retrieved_nodes = self.hybrid_retriever.retrieve(query, top_k=depth, filters=filters)
            if not retrieved_nodes:
                logging.warning("Hybrid retrieval failed. Falling back to vector retrieval.")
                retrieved_nodes = self.retriever.retrieve(query, top_k=depth, filters=filters)
        else:
            retrieved_nodes = self.retriever.retrieve(query, top_k=depth, filters=filters)

        docs_as_objects = self._format_nodes_to_docs(retrieved_nodes)

        retrieval_time = time.time() - retrieval_start
        serializable_docs = [
            {"text": doc.get("text"), "metadata": doc.get("metadata"), "score": doc.get("score")}
            for doc in docs_as_objects
        ]
        logging.info(f"RETRIEVED_CHUNKS: {json.dumps(serializable_docs, indent=2)}")
        logging.info(
            f"Retrieval time: {retrieval_time:.4f}s, Candidates: {len(docs_as_objects)}, Strategy: {retrieval_strategy}")
        return {"docs": docs_as_objects, "time": retrieval_time}

    def _rerank_docs(self, query: str, docs: List[Dict[str, Any]], strategy: Dict[str, Any]) -> Dict:
        if not strategy.get("use_reranker") or not docs:
            return {"docs": docs, "time": 0}

        rerank_start = time.time()

        # Convert dicts back to NodeWithScore objects for the reranker
        nodes_to_rerank = [
            NodeWithScore(
                node=TextNode(text=doc.get("text", ""), metadata=doc.get("metadata", {})),
                score=doc.get("score")
            )
            for doc in docs
        ]

        reranked_nodes = self.reranker.rerank_nodes(query, nodes_to_rerank)

        # Truncate the results to the top N after reranking
        top_n = strategy.get("reranker_top_n", settings.RERANKER_TOP_N)
        reranked_nodes = reranked_nodes[:top_n]

        # Convert reranked nodes back to the standardized dictionary format
        reranked_docs = self._format_nodes_to_docs(reranked_nodes)

        rerank_time = time.time() - rerank_start
        logging.info(f"Rerank time: {rerank_time:.4f}s")
        return {"docs": reranked_docs, "time": rerank_time}

    async def _decide_and_expand(self, query: str, query_analysis: Dict, initial_docs: List[Dict[str, Any]],
                           reranked_docs: Optional[List[Dict[str, Any]]], initial_filters: Dict = None) -> Dict:
        strategy = query_analysis['strategy']
        query_metadata = query_analysis['metadata']
        docs_for_confidence = reranked_docs if reranked_docs is not None and strategy.get("use_reranker") else initial_docs

        # First, calculate context confidence
        context_confidence = self.confidence_engine.calculate_context_confidence(
            query_metadata=query_metadata,
            retrieved_docs=initial_docs,
            reranked_docs=reranked_docs
        )

        # Decide initial action based on context
        action = self.confidence_engine.decide_action_from_context(context_confidence)

        final_docs = docs_for_confidence
        expansion_details = {"expanded": False, "queries": [query], "retrieval_time": 0}
        confidence_before_expansion = context_confidence.get("context_score", 0.0)

        if action == "expand":
            logging.info("Executing 'expand' action: Expanding query and re-retrieving.")
            expansion_details["expanded"] = True

            expanded_queries = await self.query_expander.expand(query, query_metadata)
            expansion_details["queries"] = expanded_queries

            expand_retrieval_start = time.time()
            all_expanded_docs = []
            for eq in expanded_queries:
                if eq != query:
                    retrieval_result = self._retrieve_docs(eq, strategy, filters=initial_filters)
                    all_expanded_docs.extend(retrieval_result["docs"])
            expansion_details["retrieval_time"] = time.time() - expand_retrieval_start

            merged_docs = self._merge_and_deduplicate(initial_docs + all_expanded_docs)

            if strategy.get("use_reranker"):
                final_rerank_result = self._rerank_docs(query, merged_docs, strategy)
                final_docs = final_rerank_result["docs"]
            else:
                final_docs = merged_docs

            # After expansion, re-evaluate context confidence and make a new decision (no more expansion)
            context_confidence = self.confidence_engine.calculate_context_confidence(
                query_metadata=query_metadata,
                retrieved_docs=merged_docs,
                reranked_docs=final_docs
            )
            action = self.confidence_engine.decide_action_from_context(context_confidence, is_post_expansion=True)
            logger.info(f"Post-expansion action decided: '{action}'")

        return {
            "action": action,
            "final_docs": final_docs,
            "confidence": context_confidence, # Pass the final context confidence
            "expansion_details": expansion_details,
            "confidence_before_expansion": confidence_before_expansion
        }

    def _merge_and_deduplicate(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique_docs = {}
        for doc in docs:
            # Use a unique identifier from metadata if available, otherwise use text
            unique_id = doc.get("metadata", {}).get("chunk_id") or doc.get("text")
            if unique_id and unique_id not in unique_docs:
                unique_docs[unique_id] = doc

        logging.info(f"Merged and deduplicated docs down to {len(unique_docs)} unique docs.")
        return list(unique_docs.values())

    def _construct_context(self, docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.get("text", "")
            metadata = doc.get("metadata", {})
            content_type = metadata.get("content_type", "text")

            part = f"--- START OF CONTEXT {i+1} ---\n"
            part += f"Source File: {metadata.get('file_name', 'Unknown')}\n"

            if content_type == "table":
                part += "Content Type: Table\n"
                part += "--- TABLE START ---\n"
                part += content
                part += "\n--- TABLE END ---\n"
            elif content_type == "code":
                language = metadata.get("language", "text")
                part += f"Content Type: Code Block ({language})\n"
                part += f"```{language}\n"
                part += content
                part += f"\n```\n"
            else: # Default to text
                part += "Content Type: Text\n"
                part += content

            part += f"--- END OF CONTEXT {i+1} ---"
            context_parts.append(part)

        return "\n\n".join(context_parts)

    def _format_sources(self, docs: List[Dict[str, Any]]) -> List[Dict]:
        sources = []
        for doc in docs:
            metadata = doc.get("metadata", {})
            source_item = {
                "file": metadata.get("file_name", "Unknown"),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
            }
            if "score" in doc and doc["score"] is not None:
                source_item["vector_score"] = float(doc["score"])

            # The reranker now updates the score in place.
            # We can add a rank if needed.
            sources.append(source_item)
        return sources

    def _format_empty_response(self, query: str, start_time: float, query_analysis: Dict) -> Dict[str, Any]:
        return {
            "answer": "Could not find any relevant information.",
            "sources": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_docs": [],
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_analysis.get('metadata', {}),
            "strategy": query_analysis.get('strategy', {}),
            "pipeline": "accurate",
        }

    def _format_abstain_response(self, decision_result: Dict, start_time: float, query_analysis: Dict) -> Dict[str, Any]:
        return {
            "answer": "Abstained from answering due to low confidence.",
            "sources": self._format_sources(decision_result.get("final_docs", [])),
            "retrieved_docs": [], # Or potentially populate from decision_result if available
            "reranked_docs": [], # Or potentially populate
            "final_docs": decision_result.get("final_docs", []),
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_analysis.get('metadata', {}),
            "strategy": query_analysis.get('strategy', {}),
            "pipeline": "accurate",
            "confidence": decision_result["confidence"],
            "action_taken": "abstain"
        }

    def _format_response(self, **kwargs) -> Dict[str, Any]:
        expansion_details = kwargs.get("expansion_details", {})
        if expansion_details.get("expanded"):
            # Add the pre-expansion confidence score to the details object
            expansion_details["confidence_before_expansion"] = kwargs.get("confidence_before_expansion", 0.0)

        confidence = kwargs.get("confidence", {})
        confidence_score = confidence.get("context_score") if isinstance(confidence, dict) else None
        query_metadata = kwargs.get("query_metadata", {})

        return {
            "answer": kwargs.get("answer"),
            "sources": kwargs.get("sources"),
            "retrieved_docs": kwargs.get("retrieved_docs", []),
            "reranked_docs": kwargs.get("reranked_docs", []),
            "final_docs": kwargs.get("final_docs", []),
            "context": kwargs.get("context", ""),
            "latency": kwargs.get("latency"),
            "query_metadata": query_metadata,
            "strategy": kwargs.get("strategy", {}),
            "query_type": query_metadata.get("query_type", "unknown"), # For convenience
            "pipeline": kwargs.get("pipeline"),
            "confidence_score": confidence_score,
            "expansion_details": expansion_details
        }

    def _format_error_response(self, error_message: str, start_time: float, query_analysis: Dict) -> Dict[str, Any]:
        return {
            "answer": f"An error occurred: {error_message}",
            "sources": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_docs": [],
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_analysis.get('metadata', {}),
            "strategy": query_analysis.get('strategy', {}),
            "pipeline": "accurate",
        }