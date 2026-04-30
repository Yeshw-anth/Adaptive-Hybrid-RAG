import time
from src.core.logging_config import logger
import uuid
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from src.config import settings
from src.core.strategy.query_analyzer import QueryAnalyzer
from src.core.strategy.router import StrategyRouter
from src.core.strategy.query_expander import QueryExpander
from src.core.decision.confidence_engine import ConfidenceEngine
from src.core.llm.ollama_client import OllamaClient as LLMClient
from src.core.pipelines.base import Pipeline
from src.core.outputlogs.output_logger import OutputLogger
from src.data.schemas import OutputLog, Document, QueryMetadata, Strategy
from src.core.caching.response_cache import ResponseCache
from src.core.query_processor import QueryProcessor
# from src.experimental.feedback.feedback_store import FeedbackStore


class RAGOrchestrator:
    """
    Orchestrator for the Adaptive RAG system.
    This class manages the entire lifecycle of a query, from analysis to 
    response generation and verification.
    """

    def __init__(self, query_analyzer: QueryAnalyzer, strategy_router: StrategyRouter, query_expander: QueryExpander, pipelines: Dict[str, Pipeline], confidence_engine: ConfidenceEngine, llm_client: LLMClient, retriever: Any = None, ingestion_pipeline: Any = None, vector_index: Any = None, hybrid_retriever: Any = None, keyword_retriever: Any = None, response_cache: Optional[ResponseCache] = None):
        self.query_analyzer: QueryAnalyzer = query_analyzer
        self.strategy_router: StrategyRouter = strategy_router
        self.query_expander: QueryExpander = query_expander
        self.pipelines: Dict[str, Pipeline] = pipelines
        self.confidence_engine: ConfidenceEngine = confidence_engine
        self.llm_client: LLMClient = llm_client
        self.retriever: Any = retriever # For parent fetching
        self.ingestion_pipeline: Any = ingestion_pipeline
        self.vector_index = vector_index
        self.hybrid_retriever = hybrid_retriever
        self.keyword_retriever = keyword_retriever
        self.output_logger = OutputLogger()
        self.cache = response_cache if response_cache else ResponseCache()
        self.query_processor = QueryProcessor()
        self.index_version = str(uuid.uuid4()) # Initial version of the index
        # self.feedback_store = FeedbackStore() # Temporarily disabled
        logger.info(f"Initialized RAG orchestrator with pipelines: {list(self.pipelines.keys())}")
        logger.info(f"Initial index version: {self.index_version}")

    def load_and_sync_retrievers(self):
        """
        This method should be called after the orchestrator is initialized,
        once the vector index has loaded its data. It ensures the hybrid
        and keyword retrievers are synced with the full document set.
        """
        self.update_retriever_nodes()

    async def ingest_and_process(self, file_path: str) -> List[Dict[str, Any]]:
        """Runs the ingestion pipeline and updates the system with new nodes."""
        logger.info(f"--- Starting ingestion process for file: {file_path} ---")
        if not self.ingestion_pipeline:
            raise ValueError("Ingestion pipeline is not configured.")

        # Run the ingestion pipeline to get the new nodes
        new_nodes = await self.ingestion_pipeline.ingest_file(file_path)
        if not new_nodes:
            logger.warning(f"Ingestion of {file_path} resulted in 0 nodes.")
            return []

        # Add the new nodes to the vector index
        if self.vector_index:
            self.vector_index.insert_nodes(new_nodes)
            logger.info(f"Inserted {len(new_nodes)} new nodes into the vector index.")
            # Persist the changes to disk
            self.vector_index.storage_context.persist(persist_dir=settings.PERSIST_DIR)
            logger.info(f"Vector index persisted to {settings.PERSIST_DIR}")
        else:
            logger.error("Vector index is not available. Cannot insert new nodes.")
            return []

        # IMPORTANT: Update the retrievers that depend on the full node list
        self.update_retriever_nodes()
        
        # Invalidate cache by updating the index version
        self.index_version = str(uuid.uuid4())
        logger.info(f"New documents ingested. Index version updated to: {self.index_version}")
        self.cache.clear() # For simplicity, we clear the cache. A more advanced strategy could be used.


        logger.info(f"--- Ingestion process for {file_path} completed successfully. ---")
        return [node.to_dict() for node in new_nodes]

    def update_retriever_nodes(self):
        """Updates the nodes for retrievers that don't automatically sync with the index."""
        logger.info("Updating nodes for keyword and hybrid retrievers...")
        if self.vector_index and hasattr(self.vector_index, 'docstore'):
            all_nodes = list(self.vector_index.docstore.docs.values())
            if self.keyword_retriever:
                # The KeywordRetriever uses a 'set_nodes' method
                self.keyword_retriever.update_corpus([node.get_content() for node in all_nodes])
                logger.info(f"Updated KeywordRetriever with {len(all_nodes)} nodes.")
            if self.hybrid_retriever:
                self.hybrid_retriever.update_corpus(all_nodes)
                logger.info(f"Updated HybridRetriever with {len(all_nodes)} nodes.")
        else:
            logger.warning("Vector index or docstore not available. Cannot update retriever nodes.")

    async def query(self, original_query: str, model_override: str | None = None) -> Dict[str, Any]:
        """
        The main entry point for processing a user query. It preprocesses the query,
        analyzes it, selects a strategy, and executes the corresponding pipeline.
        """
        start_time = time.time()
        query_id = f"query_{int(start_time)}"
        logger.info(f"--- Orchestrating query: '{original_query}' (ID: {query_id}) ---")

        # --- Step 0: Preprocess the query ---
        normalized_query = self.query_processor.normalize(original_query)
        keyword_tokens = self.query_processor.clean_for_keywords(original_query)
        logger.info(f"Normalized query for semantic search: '{normalized_query}'")
        logger.info(f"Cleaned tokens for keyword search: {keyword_tokens}")

        try:
            # We use the normalized query for semantic analysis and strategy selection
            query_metadata, strategy = await self._analyze_and_select_strategy(
                normalized_query, keyword_tokens, model_override
            )

        except Exception as e:
            logger.warning(f"Query analysis failed: {e}. Falling back to default 'accurate' strategy.", exc_info=True)
            query_metadata = QueryMetadata(
                intent="fact-seeking",
                complexity="high",
                keywords=keyword_tokens, # Use cleaned tokens for fallback
                expected_answer_format="explanation",
                query_type="complex",
                normalized_query=normalized_query,
                keyword_tokens=keyword_tokens
            )
            strategy = self.strategy_router.select_strategy(query_metadata, model_override)

        try:
            # The pipeline will receive the original query and the processed versions within the metadata
            pipeline_result = await self._execute_pipeline(original_query, strategy, query_metadata)
            
            # --- Step 3: Finalize and Log ---
            end_time = time.time()
            latency = end_time - start_time
            pipeline_result['latency'] = latency
            
            logger.info(f"Query orchestration completed in {latency:.2f} seconds.")
            
            self._log_output(query_id, original_query, pipeline_result)
            
            return pipeline_result
        except Exception as e:
            logger.error(f"Error during pipeline execution for ID {query_id}: {e}", exc_info=True)
            end_time = time.time()
            latency = end_time - start_time
            logger.info(f"Query orchestration failed in {latency:.2f} seconds.")
            return {"error": str(e), "status": "failed", "query_id": query_id, "latency": latency}

    async def _analyze_and_select_strategy(self, normalized_query: str, keyword_tokens: List[str], model_override: str | None = None) -> Tuple[QueryMetadata, "Strategy"]:
        """Analyzes the query and selects the appropriate RAG strategy."""
        logger.info("Step 1: Analyzing query and selecting strategy.")
        query_metadata = await self.query_analyzer.analyze(
            normalized_query=normalized_query,
            keyword_tokens=keyword_tokens,
            model_override=model_override
        )
        strategy = self.strategy_router.select_strategy(query_metadata,model_override=model_override)
        logger.info(f"Strategy selected: {strategy.pipeline}")
        return query_metadata, strategy

    async def _execute_pipeline(self, query: str, strategy: Strategy, query_metadata: QueryMetadata) -> Dict[str, Any]:
        """Executes the selected pipeline with the given query and strategy."""
        logger.info(f"Step 2: Executing pipeline: {strategy.pipeline}")
        
        pipeline_name = strategy.pipeline
        selected_pipeline = self.pipelines.get(pipeline_name)
        if not selected_pipeline:
            logger.error(f"Pipeline '{pipeline_name}' not found.")
            raise ValueError(f"Pipeline '{pipeline_name}' not found.")

        # The pipeline will receive a dictionary with 'metadata' and 'strategy' keys.
        query_analysis = {
            "metadata": query_metadata,
            "strategy": strategy
        }

        pipeline_result = await selected_pipeline.execute(
            query=query, 
            query_analysis=query_analysis
        )
        
        # Post-processing for pipelines that only retrieve documents (like CodePipeline)
        if 'answer' not in pipeline_result and 'reranked_docs' in pipeline_result:
            logger.info("Pipeline returned documents. Generating final answer.")
            
            context_docs = pipeline_result['reranked_docs']
            context_str = "\n\n".join(
                [f"Source {i+1}: {doc.text}" for i, doc in enumerate(context_docs)]
            )
            
            pipeline_result['answer'] = await self.llm_client.generate_structured_response(
                context=context_str,
                query=query,
                model=strategy.model
            )
            pipeline_result['context'] = context_str
            pipeline_result['final_docs'] = context_docs

        # Ensure metadata and strategy are in the final result for logging
        pipeline_result['query_metadata'] = query_metadata
        pipeline_result['strategy'] = strategy
        
        return pipeline_result

    def _log_output(self, query_id: str, query: str, result: Dict):
        """
        Logs the result of a query for later evaluation using the FeedbackStore.
        This method is defensive and uses .get() to avoid KeyErrors
        and aligns with the OutputLog Pydantic schema.
        """
        try:
            # The pipeline result might contain a 'confidence' dictionary (from accurate_pipeline)
            # or a flat 'confidence_score' (from fast_pipeline). This handles both cases.
            confidence_info = result.get("confidence", {})
            if isinstance(confidence_info, dict):
                confidence_score = confidence_info.get("final_score")
                action_taken = confidence_info.get("final_action")
            else:
                confidence_score = result.get("confidence_score")
                action_taken = result.get("action_taken")

            query_metadata_obj = result.get("query_metadata")
            strategy_obj = result.get("strategy")

            log_entry = OutputLog(
                query=query,
                query_metadata=query_metadata_obj if query_metadata_obj else {},
                selected_strategy=strategy_obj if strategy_obj else {},
                final_answer=result.get("answer", ""),
                final_context=result.get("context", ""),
                retrieved_docs=result.get("retrieved_docs", []),
                reranked_docs=result.get("reranked_docs"),
                final_docs=result.get("final_docs", []),
                latency=result.get("latency", 0.0),
                confidence_score=confidence_score,
                action_taken=action_taken,
                expansion_triggered=result.get("expansion_details", {}).get("expanded", False),
                timestamp=datetime.utcnow().isoformat()
            )
            # Use the FeedbackStore to log the evaluation - Temporarily disabled
            # self.feedback_store.log_feedback(query_id=query_id, evaluation=log_entry.model_dump())
            self.output_logger.log(log_entry)
        except Exception as e:
            # This will catch PydanticValidationErrors and other issues
            logger.error(f"Failed to create or log outputlogs data for query_id {query_id}: {e}", exc_info=True)