import logging
import faiss
import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore

from src.config import settings
from src.data.embedding.embedder import Embedder
from src.core.llm.ollama_client import OllamaClient
from src.core.retrieval.retriever import Retriever
from src.core.retrieval.cross_encoder import CrossEncoderReranker
from src.core.retrieval.keyword_retriever import KeywordRetriever
from src.core.strategy.query_analyzer import QueryAnalyzer
from src.core.strategy.query_expander import QueryExpander
from src.core.strategy.router import StrategyRouter
from src.core.decision.cost_latency_controller import CostLatencyController
from src.core.decision.confidence_engine import ConfidenceEngine
from src.core.retrieval.hybrid_retriever import HybridRetriever
from src.core.pipelines.fast_pipeline import FastPipeline
from src.core.pipelines.accurate_pipeline import AccuratePipeline
from src.core.pipelines.code_pipeline import CodePipeline
from src.core.pipelines.structured_pipeline import StructuredPipeline
from src.core.pipelines.keyword_pipeline import KeywordPipeline
from src.core.ingestion import IngestionPipeline
from src.chunking.chunking_engine import ChunkingEngine
from src.chunking.structure_segmenter import StructuralSegmenter
from src.chunking.semantic_segmenter import SemanticSegmenter
from src.chunking.segmentation_evaluator import SegmentationEvaluator
from src.chunking.strategy_factory import StrategyFactory
from src.chunking.strategies.text_strategy import TextChunkingStrategy
from src.core.caching.response_cache import ResponseCache
from src.core.orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)


class SystemBuilder:
    """
    A builder class responsible for instantiating and wiring all components
    of the adaptive RAG system based on configuration.
    """
    def __init__(self):
        self.components = {}

    def build_all(self):
        """Builds all components and the final RAG pipeline."""
        logger.info("--- System Build Process Started ---")
        self._build_core_services()
        self._build_rag_components()
        self._build_pipelines()
        self._build_orchestrator()
        logger.info("--- System Build Process Finished ---")
        return (
            self.components['rag_orchestrator'], 
            self.components['vector_index'],
            self.components['chunking_engine']
        )

    def _build_core_services(self):
        logger.info("Building core services...")
        embedder = Embedder(model_name=settings.EMBED_MODEL_NAME)
        llm_client = OllamaClient()
        response_cache = ResponseCache()
        
        # Respect the CLEAR_ON_RESTART setting for the persistent vector store.
        if settings.CLEAR_ON_RESTART and os.path.exists(settings.PERSIST_DIR):
            logger.warning("CLEAR_ON_RESTART is True. Deleting existing vector store.")
            import shutil
            shutil.rmtree(settings.PERSIST_DIR)

        if os.path.exists(settings.PERSIST_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=settings.PERSIST_DIR)
            vector_index = load_index_from_storage(storage_context, embed_model=embedder)
        else:
            d = settings.EMBEDDING_DIM
            faiss_index = faiss.IndexFlatIP(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index = VectorStoreIndex.from_documents([], storage_context=storage_context, embed_model=embedder)

        self.components['embedder'] = embedder
        self.components['llm_client'] = llm_client
        self.components['vector_index'] = vector_index
        self.components['response_cache'] = response_cache
        logger.info("Core services built.")

    def _build_rag_components(self):
        logger.info("Building RAG components...")
        vector_index = self.components['vector_index']
        llm_client = self.components['llm_client']
        embedder = self.components['embedder']
        all_docs = list(vector_index.docstore.docs.values())

        retriever = Retriever(vector_index=vector_index)
        keyword_retriever = KeywordRetriever(nodes=all_docs)
        reranker = CrossEncoderReranker()
        query_analyzer = QueryAnalyzer(llm_wrapper=llm_client)
        query_expander = QueryExpander(llm_wrapper=llm_client)
        confidence_engine = ConfidenceEngine(llm_wrapper=llm_client)
        hybrid_retriever = HybridRetriever(vector_retriever=retriever, all_docs=all_docs)
        cost_latency_controller = CostLatencyController()
        strategy_router = StrategyRouter(cost_latency_controller=cost_latency_controller)

        # Build the advanced segmentation and chunking modules
        structural_segmenter = StructuralSegmenter(embedder=embedder)
        semantic_segmenter = SemanticSegmenter()
        
        segmentation_evaluator = SegmentationEvaluator(
            structural_segmenter=structural_segmenter, 
            semantic_segmenter=semantic_segmenter
        )

        strategy_factory = StrategyFactory(embedding_model=embedder)
        chunking_engine = ChunkingEngine(
            segmentation_evaluator=segmentation_evaluator,
            strategy_factory=strategy_factory
        )
        
        ingestion_pipeline = IngestionPipeline(
            chunking_engine=chunking_engine
        )

        self.components.update({
            'retriever': retriever,
            'keyword_retriever': keyword_retriever,
            'reranker': reranker,
            'query_analyzer': query_analyzer,
            'query_expander': query_expander,
            'confidence_engine': confidence_engine,
            'hybrid_retriever': hybrid_retriever,
            'cost_latency_controller': cost_latency_controller,
            'strategy_router': strategy_router,
            'ingestion_pipeline': ingestion_pipeline,
            'chunking_engine': chunking_engine
        })
        logger.info("RAG components built.")

    def _build_pipelines(self):
        logger.info("Building pipelines...")
        c = self.components
        fast_pipeline = FastPipeline(retriever=c['retriever'], llm_client=c['llm_client'], confidence_engine=c['confidence_engine'])
        accurate_pipeline = AccuratePipeline(
            retriever=c['retriever'], hybrid_retriever=c['hybrid_retriever'], reranker=c['reranker'],
            llm_client=c['llm_client'], query_expander=c['query_expander'],
            confidence_engine=c['confidence_engine']
        )
        code_pipeline = CodePipeline(
            retriever=c['hybrid_retriever'], reranker=c['reranker']
        )
        structured_pipeline = StructuredPipeline(llm_client=c['llm_client'], retriever=c['retriever'], confidence_engine=c['confidence_engine'])
        keyword_pipeline = KeywordPipeline(keyword_retriever=c['keyword_retriever'], llm_client=c['llm_client'])
        
        self.components['pipelines'] = {
            "fast": fast_pipeline,
            "accurate": accurate_pipeline,
            "code": code_pipeline,
            "structured": structured_pipeline,
            "keyword": keyword_pipeline
        }
        logger.info("Pipelines built.")

    def _build_orchestrator(self):
        logger.info("Building RAG orchestrator...")
        c = self.components
        rag_orchestrator = RAGOrchestrator(
            query_analyzer=c['query_analyzer'],
            strategy_router=c['strategy_router'],
            pipelines=c['pipelines'],
            query_expander=c['query_expander'],
            confidence_engine=c['confidence_engine'],
            llm_client=c['llm_client'],
            retriever=c['retriever'],
            ingestion_pipeline=c['ingestion_pipeline'],
            vector_index=c['vector_index'],
            hybrid_retriever=c['hybrid_retriever'],
            keyword_retriever=c['keyword_retriever'],
            response_cache=c['response_cache']
        )
        self.components['rag_orchestrator'] = rag_orchestrator
        logger.info("RAG orchestrator built.")