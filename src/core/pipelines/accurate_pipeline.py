# Force reload
import time
from src.core.logging_config import logger
import json
from typing import Dict, Any, List, Optional

from src.data.schemas import Document, QueryMetadata, Strategy
from src.core.pipelines.base import Pipeline
from src.core.retrieval.retriever import Retriever
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.retrieval.hybrid_graph_retriever import HybridGraphRetriever
from src.core.retrieval.cross_encoder import CrossEncoderReranker
from llama_index.core.schema import NodeWithScore, TextNode
from src.core.llm.ollama_client import OllamaClient
from src.core.strategy.query_expander import QueryExpander
from src.core.decision.confidence_engine import ConfidenceEngine
from src.core.caching.response_cache import ResponseCache

from src.config.settings import settings

class AccuratePipeline(Pipeline):
    """
    A sophisticated RAG pipeline that uses hybrid retrieval, reranking, and a
    confidence-driven expansion loop to provide high-quality answers.
    """

    def __init__(self, retriever: Retriever, hybrid_retriever: HybridRetriever, hybrid_graph_retriever: HybridGraphRetriever, reranker: CrossEncoderReranker,
                 llm_client: OllamaClient, query_expander: QueryExpander, confidence_engine: ConfidenceEngine):
        self.retriever = retriever
        self.hybrid_retriever = hybrid_retriever
        self.hybrid_graph_retriever = hybrid_graph_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.query_expander = query_expander
        self.confidence_engine = confidence_engine


    async def execute(self, query: str, query_analysis: Dict[str, Any], cache: Optional[ResponseCache] = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"--- Running AccuratePipeline for query: '{query}' ---")

        strategy: Strategy = query_analysis['strategy']
        query_metadata: QueryMetadata = query_analysis['metadata']
        chunking_strategy = strategy.chunking_strategy if hasattr(strategy, 'chunking_strategy') else 'semantic'
        logger.info(f"Using chunking strategy: {chunking_strategy}")

        logger.info("Step 1: Analyzing query for section filters...")
        filters = None

        logger.info("Step 2: Performing initial retrieval...")
        retrieval_result = await self._retrieve_docs(query, strategy, model=strategy.model, filters=filters)
        initial_nodes = retrieval_result["docs"]

        if not initial_nodes and filters:
            logger.warning("Filtered retrieval yielded no results. Falling back to broad retrieval.")
            retrieval_result = await self._retrieve_docs(query, strategy, model=strategy.model, filters=None)
            initial_nodes = retrieval_result["docs"]

        if not initial_nodes:
            logger.warning("No documents found during initial retrieval.")
            return self._format_empty_response(query, start_time, strategy, query_metadata)
        logger.info(f"Step 2: Retrieved {len(initial_nodes)} documents.")

        logger.info("Step 3: Reranking retrieved documents...")
        rerank_result = self._rerank_docs(query, initial_nodes, strategy)
        reranked_nodes = rerank_result["docs"]
        if reranked_nodes:
            logger.info(f"Step 3: Reranking complete. Top document score: {reranked_nodes[0].score}")
        else:
            logger.warning("Reranking returned no documents.")

        logger.info("Step 4: Making decision on action (expand, generate, or abstain)...")
        decision_result = await self._decide_and_expand(
            query, strategy, query_metadata, initial_nodes, reranked_nodes, initial_filters=filters
        )

        action = decision_result["action"]
        if action == "abstain":
            return self._format_abstain_response(decision_result, start_time, strategy, query_metadata)

        final_nodes = decision_result["final_docs"]
        
        # Caching logic starts here
        if cache:
            final_node_ids = sorted([node.node.id_ for node in final_nodes])
            cached_response = cache.get(query, final_node_ids)
            if cached_response:
                logger.info(f"Cache hit for query: '{query}'. Returning cached response.")
                cached_response["latency"] = time.time() - start_time
                cached_response["pipeline"] = "accurate-cached"
                return cached_response
            logger.info(f"Cache miss for query: '{query}'. Proceeding with generation.")

        final_docs_for_context = final_nodes

        if strategy.use_parent_child:
            logger.info("Step 5a: Strategy requires parent-child retrieval. Fetching parent documents.")
            final_docs_for_context = self._fetch_and_add_parents(final_nodes)

        # Convert to dicts for the final stages
        final_docs_as_dicts = self._format_nodes_to_docs(final_docs_for_context)
        
        # Use fused context if available (from hybrid_graph retriever)
        if "full_result" in retrieval_result and isinstance(retrieval_result["full_result"], dict) and "fused_context" in retrieval_result["full_result"]:
            context = retrieval_result["full_result"]["fused_context"]
        # Use direct graph context if available (from graph_native retriever)
        elif "graph_context" in retrieval_result:
            context = retrieval_result["graph_context"]
        else:
            context = self._construct_context(final_docs_as_dicts)
        
        logger.info(f"CONTEXT_SENT_TO_LLM: {context}")

        try:
            answer = await self.llm_client.generate_structured_response(context, query, strategy.model)
        except Exception as e:
            logger.error(f"Error during LLM generation in AccuratePipeline: {e}")
            return self._format_error_response(str(e), start_time, strategy, query_metadata)

        logger.info(f"LLM_RAW_RESPONSE: {answer}")


        response = self._format_response(
            answer=answer,
            sources=self._format_sources(final_docs_as_dicts),
            retrieved_docs=self._format_nodes_to_docs(initial_nodes),
            reranked_docs=self._format_nodes_to_docs(reranked_nodes if reranked_nodes else []),
            final_docs=final_docs_as_dicts,
            context=context,
            latency=time.time() - start_time,
            query_metadata=query_metadata,
            strategy=strategy,
            pipeline="accurate",
            confidence=decision_result["confidence"],
            action_taken=action,
            expansion_details=decision_result["expansion_details"],
            confidence_before_expansion=decision_result.get("confidence_before_expansion", {}).get("final_score", 0.0)
        )
        
        if cache:
            final_node_ids = sorted([node.node.id_ for node in final_nodes])
            cache.set(query, final_node_ids, response)
            logger.info(f"Response for query '{query}' stored in cache.")

        logger.info(f"GENERATED_RESPONSE: {json.dumps(response, indent=2)}")
        return response

    def _fetch_and_add_parents(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Fetches parent documents for a list of nodes and merges them.
        Returns a list of NodeWithScore objects.
        """
        if not self.retriever:
            logger.warning("Retriever not available. Skipping parent fetch.")
            return nodes

        parent_ids = set()
        for node in nodes:
            parent_id = node.metadata.get("parent_id")
            if parent_id:
                parent_ids.add(parent_id)
        
        if not parent_ids:
            logger.info("No parent IDs found in the retrieved nodes.")
            return nodes

        logger.info(f"Fetching {len(parent_ids)} parent documents.")
        
        # Retrieve parent nodes using the new retriever method
        parent_nodes_dict = self.retriever.get_nodes_by_ids(list(parent_ids))
        
        # Create NodeWithScore objects for the parents. Assign a neutral score.
        parent_node_with_scores = [
            NodeWithScore(node=parent_node, score=0.0) for parent_node in parent_nodes_dict.values()
        ]
        
        logger.info(f"Successfully fetched {len(parent_node_with_scores)} parent nodes.")

        # Merge original nodes with parent nodes, preserving order and prioritizing original nodes
        combined_nodes = self._merge_and_deduplicate(nodes + parent_node_with_scores)
        return combined_nodes


    async def _retrieve_docs(self, query: str, strategy: "Strategy", model: str, filters: Dict[str, str] = None) -> Dict:
        retrieval_start = time.time()
        retrieval_strategy = strategy.retrieval_strategy
        depth = strategy.top_k
        retrieval_result = None

        if retrieval_strategy == 'graph_native':
            from src.core.retrieval.graph_retriever import GraphRetriever
            graph_retriever = GraphRetriever()
            graph_context = await graph_retriever.retrieve(query)
            # For graph_native, the "context" is the result, and docs are empty
            return {"docs": [], "time": time.time() - retrieval_start, "graph_context": graph_context}

        if retrieval_strategy == 'hybrid':
            retrieved_nodes = self.hybrid_retriever.retrieve(query, top_k=depth, filters=filters)
            if not retrieved_nodes:
               logger.warning("Hybrid retrieval failed. Falling back to vector retrieval.")

        elif retrieval_strategy == 'hybrid_graph':
            retrieval_result = await self.hybrid_graph_retriever.retrieve(query, top_k=depth)
            retrieved_nodes = retrieval_result
            # The graph context is handled separately in the main execute method
        else:
            retrieved_nodes = self.retriever.retrieve(query, top_k=depth, filters=filters)

        retrieval_time = time.time() - retrieval_start
        logger.info(
            f"Retrieval time: {retrieval_time:.4f}s, Candidates: {len(retrieved_nodes)}, Strategy: {retrieval_strategy}")
        
        # For hybrid_graph, we need to pass the full result
        if retrieval_strategy == 'hybrid_graph':
            return {"docs": retrieved_nodes, "time": retrieval_time, "full_result": retrieval_result}
            
        return {"docs": retrieved_nodes, "time": retrieval_time}

    def _rerank_docs(self, query: str, docs: List[NodeWithScore], strategy: "Strategy") -> Dict:
        if not strategy.use_reranker or not docs:
            return {"docs": docs, "time": 0}

        rerank_start = time.time()

        # Rerank the nodes
        reranked_nodes = self.reranker.rerank_nodes(query, docs)

        # Truncate the results to the top N after reranking
        top_n = strategy.reranker_top_n if hasattr(strategy, 'reranker_top_n') else settings.RERANKER_TOP_N
        reranked_nodes = reranked_nodes[:top_n]

        rerank_time = time.time() - rerank_start
        logger.info(f"Rerank time: {rerank_time:.4f}s")
        return {"docs": reranked_nodes, "time": rerank_time}

    async def _decide_and_expand(self, query: str, strategy: Strategy, query_metadata: QueryMetadata, initial_nodes: List[NodeWithScore],
                               reranked_nodes: Optional[List[NodeWithScore]], initial_filters: Dict = None) -> Dict:
        
        docs_for_confidence = reranked_nodes if reranked_nodes is not None and strategy.use_reranker else initial_nodes

        context_confidence = self.confidence_engine.calculate_context_confidence(
            query_metadata=query_metadata,
            retrieved_docs=initial_nodes,
            reranked_docs=reranked_nodes
        )
        action = self.confidence_engine.decide_action_from_context(context_confidence)

        final_nodes = docs_for_confidence
        expansion_details = {"expanded": False, "queries": [query], "retrieval_time": 0.0}
        confidence_before_expansion = context_confidence

        if action == "expand":
            logger.info("Executing 'expand' action: Expanding query and re-retrieving.")
            expansion_details["expanded"] = True

            expanded_queries = await self.query_expander.expand(query, query_metadata)
            expansion_details["queries"] = expanded_queries

            expand_retrieval_start = time.time()
            all_expanded_nodes = []
            for eq in expanded_queries:
                if eq != query:
                    retrieval_result = await self._retrieve_docs(eq, strategy, model=strategy.model, filters=initial_filters)
                    all_expanded_nodes.extend(retrieval_result["docs"])
            expansion_details["retrieval_time"] = time.time() - expand_retrieval_start

            merged_nodes = self._merge_and_deduplicate(initial_nodes + all_expanded_nodes)

            if strategy.use_reranker:
                final_rerank_result = self._rerank_docs(query, merged_nodes, strategy)
                final_nodes = final_rerank_result["docs"]
            else:
                final_nodes = merged_nodes

            # Re-evaluate confidence post-expansion
            context_confidence = self.confidence_engine.calculate_context_confidence(
                query_metadata=query_metadata.model_dump(),
                retrieved_docs=merged_nodes,
                reranked_docs=final_nodes
            )
            action = self.confidence_engine.decide_action_from_context(context_confidence, is_post_expansion=True)
            logger.info(f"Post-expansion action decided: '{action}'")

        return {
            "action": action,
            "final_docs": final_nodes,
            "confidence": context_confidence,
            "expansion_details": expansion_details,
            "confidence_before_expansion": confidence_before_expansion
        }

    def _merge_and_deduplicate(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Merges and deduplicates a list of NodeWithScore objects based on node ID."""
        unique_nodes = {}
        for node in nodes:
            if node.node.id_ not in unique_nodes:
                unique_nodes[node.node.id_] = node
        
        logger.info(f"Merged and deduplicated docs down to {len(unique_nodes)} unique docs.")
        return list(unique_nodes.values())

    def _format_nodes_to_docs(self, nodes: Optional[List[NodeWithScore]]) -> List[Dict[str, Any]]:
        """Converts a list of NodeWithScore to a list of standardized dictionaries."""
        if not nodes:
            return []
        
        formatted_docs = []
        for node in nodes:
            doc = {
                "id": node.node.id_, # Capture the node's unique ID
                "text": node.get_text(),
                "metadata": node.metadata or {},
                "score": node.score,
                "file_path": node.metadata.get("file_path", "Unknown") # Capture the file path
            }
            formatted_docs.append(doc)
        return formatted_docs

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
            source_item: Dict[str, Any] = {
                "file": doc.get("file_path", "Unknown"),
                "chunk_id": doc.get("id", "Unknown"),
            }
            if "score" in doc and doc["score"] is not None:
                source_item["vector_score"] = float(doc["score"])

            sources.append(source_item)
        return sources

    def _format_empty_response(self, query: str, start_time: float, strategy: Strategy, query_metadata: QueryMetadata) -> Dict[str, Any]:
        return {
            "answer": "Could not find any relevant information.",
            "sources": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_docs": [],
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_metadata.model_dump(),
            "strategy": strategy.model_dump(),
            "pipeline": "accurate",
        }

    def _format_abstain_response(self, decision_result: Dict, start_time: float, strategy: Strategy, query_metadata: QueryMetadata) -> Dict[str, Any]:
        return {
            "answer": "Abstained from answering due to low confidence.",
            "sources": self._format_sources(decision_result.get("final_docs", [])),
            "retrieved_docs": [], # Or potentially populate from decision_result if available
            "reranked_docs": [], # Or potentially populate
            "final_docs": decision_result.get("final_docs", []),
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_metadata.model_dump(),
            "strategy": strategy.model_dump(),
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
        query_metadata: Optional[QueryMetadata] = kwargs.get("query_metadata")
        strategy: Optional[Strategy] = kwargs.get("strategy")

        raw_answer = kwargs.get("answer", {})
        answer_content = raw_answer.get("content", "") if isinstance(raw_answer, dict) else raw_answer

        return {
            "answer": answer_content,
            "sources": kwargs.get("sources"),
            "retrieved_docs": kwargs.get("retrieved_docs", []),
            "reranked_docs": kwargs.get("reranked_docs", []),
            "final_docs": kwargs.get("final_docs", []),
            "context": kwargs.get("context", ""),
            "latency": kwargs.get("latency"),
            "query_metadata": query_metadata.model_dump() if query_metadata else {},
            "strategy": strategy.model_dump() if strategy else {},
            "query_type": query_metadata.query_type if query_metadata else "unknown", # For convenience
            "pipeline": kwargs.get("pipeline"),
            "confidence_score": confidence_score,
            "expansion_details": expansion_details
        }

    def _format_error_response(self, error_message: str, start_time: float, strategy: Strategy, query_metadata: QueryMetadata) -> Dict[str, Any]:
        return {
            "answer": f"An error occurred: {error_message}",
            "sources": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_docs": [],
            "context": "",
            "latency": time.time() - start_time,
            "query_metadata": query_metadata.model_dump(),
            "strategy": strategy.model_dump(),
            "pipeline": "accurate",
        }