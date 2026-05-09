from src.core.logging_config import logger
import os
import shutil
import faiss
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore

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
from src.core.ingestion import IngestionPipeline, IngestionRouter
from src.chunking.chunking_engine import ChunkingEngine
from src.chunking.structure_segmenter import StructuralSegmenter
from src.chunking.semantic_segmenter import SemanticSegmenter
from src.chunking.segmentation_evaluator import SegmentationEvaluator
from src.chunking.strategy_factory import StrategyFactory
from src.chunking.strategies.text_strategy import TextChunkingStrategy
from src.chunking.strategies.code_strategy import CodeChunkingStrategy
from src.chunking.strategies.markdown_strategy import MarkdownChunkingStrategy
from src.chunking.strategies.image_strategy import ImageChunkingStrategy
from src.chunking.metadata_enricher import MetadataEnricher
from src.core.caching.response_cache import ResponseCache
from src.core.orchestrator import RAGOrchestrator
from src.core.graph.graph_builder import KnowledgeGraphBuilder
from src.core.retrieval.graph_retriever import GraphRetriever
from src.core.graph.graph_store import NetworkxGraphStore
from src.core.retrieval.hybrid_graph_retriever import HybridGraphRetriever


class SystemBuilder:
    """
    A builder class responsible for instantiating and wiring all components
    of the adaptive RAG system based on configuration.
    """
    def __init__(self, settings):
        self.settings = settings
        self.components = {}

    def shutdown(self):
        """Handles graceful shutdown of components, like persisting indexes, using atomic operations."""
        logger.info("--- Starting System Shutdown ---")

        # 1. Persist the vector index atomically
        vector_index = self.components.get('vector_index')
        if vector_index:
            persist_dir = self.settings.VECTOR_STORE_PATH
            temp_dir = persist_dir.with_suffix('.tmp')
            logger.info(f"Attempting to atomically persist index to: {persist_dir}")

            try:
                # Step 1: Persist to a temporary directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir) # Ensure clean state
                vector_index.storage_context.persist(persist_dir=str(temp_dir))
                logger.info(f"Successfully persisted index to temporary directory: {temp_dir}")

                # Step 2: Atomically replace the old directory with the new one
                if persist_dir.exists():
                    shutil.rmtree(persist_dir)
                
                os.rename(temp_dir, persist_dir)
                
                logger.info(f"Successfully and atomically persisted index to {persist_dir}")

            except Exception as e:
                logger.error(f"Failed to atomically persist vector index: {e}", exc_info=True)
                # If anything goes wrong, try to clean up the temporary directory
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        logger.warning(f"Cleaned up failed temporary persistence directory: {temp_dir}")
                    except Exception as cleanup_e:
                        logger.error(f"FATAL: Failed to clean up temporary directory {temp_dir}: {cleanup_e}")

        # 2. Persist the graph store (which now has its own atomic save)
        graph_store = self.components.get('graph_store')
        if graph_store:
            graph_store.shutdown()
            
        logger.info("--- System Shutdown Complete ---")

    def build_all(self):
        """Builds all components of the RAG system in the correct order."""
        logger.info("--- Starting System Build ---")
        self._build_core_infrastructure()
        self._build_rag_components()
        self._build_pipelines()
        self._build_orchestrator()
        self.components['rag_orchestrator'].load_and_sync_retrievers()
        logger.info("--- System Build Complete ---")
        
        return (
            self.components.get('rag_orchestrator'),
            self.components.get('vector_index'),
            self.components.get('ingestion_pipeline')
        )

    def _build_core_infrastructure(self):
        """Initializes core components like LLM clients and data stores."""
        logger.info("Building core infrastructure...")
        
        # Ensure base directories exist
        self.settings.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self.settings.CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.settings.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
        self.settings.IMAGE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        llm_client = OllamaClient()
        embedder = Embedder(model_name=self.settings.EMBED_MODEL_NAME)
        graph_store = NetworkxGraphStore(graph_path=self.settings.GRAPH_FILE_PATH)

        try:
            logger.info(f"Attempting to load existing index from: {self.settings.VECTOR_STORE_PATH}")
            if not any(self.settings.VECTOR_STORE_PATH.iterdir()):
                raise ValueError("Storage directory is empty.")

            storage_context = StorageContext.from_defaults(persist_dir=str(self.settings.VECTOR_STORE_PATH))
            vector_store = FaissVectorStore.from_persist_dir(str(self.settings.VECTOR_STORE_PATH))
            
            vector_index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=embedder
            )
            faiss_index = vector_store.client
            logger.info(f"Successfully loaded existing index with {faiss_index.ntotal} vectors.")

        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Failed to load existing index ({e}). Creating a new one.")
            faiss_index = faiss.IndexFlatL2(self.settings.EMBEDDING_DIM)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context,
                embed_model=embedder
            )
            logger.info("Created a new, empty index.")

        self.components['faiss_index'] = faiss_index
        self.components['vector_store'] = vector_store
        self.components['storage_context'] = storage_context
        
        self.components.update({
            'llm_client': llm_client,
            'embedder': embedder,
            'vector_store': vector_store,
            'vector_index': vector_index,
            'graph_store': graph_store
        })
        logger.info("Core infrastructure built.")

    def _build_rag_components(self):
        """Initializes RAG-specific components like retrievers and engines."""
        logger.info("Building RAG components...")
        c = self.components
        
        # Retrievers
        retriever = Retriever(c['vector_index'])
        all_docs = list(c['vector_index'].docstore.docs.values())
        keyword_retriever = KeywordRetriever(all_docs)
        hybrid_retriever = HybridRetriever(retriever, all_docs)
        graph_retriever = GraphRetriever()
        hybrid_graph_retriever = HybridGraphRetriever(retriever, graph_retriever)

        # Reranker
        reranker = CrossEncoderReranker()

        # Query Processors
        query_analyzer = QueryAnalyzer(c['llm_client'], self.settings.DEFAULT_LLM_MODEL)
        query_expander = QueryExpander(c['llm_client'])
        cost_latency_controller = CostLatencyController()
        strategy_router = StrategyRouter(cost_latency_controller)

        # Engines
        confidence_engine = ConfidenceEngine(c['llm_client'])
        chunking_engine = self._build_chunking_engine()

        # Ingestion
        ingestion_pipeline = IngestionPipeline(
            chunking_engine=chunking_engine,
            graph_builder=KnowledgeGraphBuilder(llm_client=c['llm_client'], graph_store=c['graph_store']),
            llm_client=c['llm_client'],
            graph_store=c['graph_store'],
            vector_index=c['vector_index']
        )

        self.components.update({
            'retriever': retriever,
            'keyword_retriever': keyword_retriever,
            'hybrid_retriever': hybrid_retriever,
            'graph_retriever': graph_retriever,
            'hybrid_graph_retriever': hybrid_graph_retriever,
            'reranker': reranker,
            'query_analyzer': query_analyzer,
            'query_expander': query_expander,
            'strategy_router': strategy_router,
            'confidence_engine': confidence_engine,
            'chunking_engine': chunking_engine,
            'ingestion_pipeline': ingestion_pipeline
        })
        logger.info("RAG components built.")

    def _build_chunking_engine(self):
        """Builds the adaptive chunking engine with multiple strategies."""
        logger.info("Building chunking engine...")
        structural_segmenter = StructuralSegmenter(self.components['embedder'])
        semantic_segmenter = SemanticSegmenter(model_name=self.settings.EMBED_MODEL_NAME)

        strategies = {
            "text": TextChunkingStrategy(embedding_model=self.components['embedder']),
            "code": CodeChunkingStrategy(),
            "markdown": MarkdownChunkingStrategy(),
            "image": ImageChunkingStrategy()
        }

        strategy_factory = StrategyFactory(strategies)
        segmentation_evaluator = SegmentationEvaluator(
            structural_segmenter=structural_segmenter,
            semantic_segmenter=semantic_segmenter
        )
        metadata_enricher = MetadataEnricher()
        
        return ChunkingEngine(
            segmentation_evaluator=segmentation_evaluator,
            strategy_factory=strategy_factory
        )

    def _build_pipelines(self):
        """Initializes the various RAG pipelines."""
        logger.info("Building RAG pipelines...")
        c = self.components
        
        fast_pipeline = FastPipeline(
            retriever=c['retriever'],
            graph_retriever=c['graph_retriever'],
            llm_client=c['llm_client'],
            confidence_engine=c['confidence_engine']
        )
        
        accurate_pipeline = AccuratePipeline(
            retriever=c['retriever'],
            hybrid_retriever=c['hybrid_retriever'],
            hybrid_graph_retriever=c['hybrid_graph_retriever'],
            reranker=c['reranker'],
            llm_client=c['llm_client'],
            query_expander=c['query_expander'],
            confidence_engine=c['confidence_engine']
        )
        
        keyword_pipeline = KeywordPipeline(
            keyword_retriever=c['keyword_retriever'],
            llm_client=c['llm_client']
        )
        
        code_pipeline = CodePipeline(
            retriever=c['hybrid_retriever'],
            reranker=c['reranker']
        )
        
        structured_pipeline = StructuredPipeline(
            retriever=c['retriever'],
            llm_client=c['llm_client'],
            confidence_engine=c['confidence_engine']
        )

        self.components.update({
            'fast_pipeline': fast_pipeline,
            'accurate_pipeline': accurate_pipeline,
            'keyword_pipeline': keyword_pipeline,
            'code_pipeline': code_pipeline,
            'structured_pipeline': structured_pipeline
        })
        logger.info("RAG pipelines built.")

    def _build_orchestrator(self):
        """Initializes the main RAG orchestrator."""
        logger.info("Building RAG orchestrator...")
        c = self.components
        
        pipelines = {
            "fast": c['fast_pipeline'],
            "accurate": c['accurate_pipeline'],
            "keyword": c['keyword_pipeline'],
            "code": c['code_pipeline'],
            "structured": c['structured_pipeline']
        }
        
        rag_orchestrator = RAGOrchestrator(
            query_analyzer=c['query_analyzer'],
            strategy_router=c['strategy_router'],
            query_expander=c['query_expander'],
            pipelines=pipelines,
            confidence_engine=c['confidence_engine'],
            llm_client=c['llm_client'],
            retriever=c['retriever'],
            ingestion_pipeline=self.components.get('ingestion_pipeline'),
            vector_index=c['vector_index'],
            hybrid_retriever=c['hybrid_retriever'],
            keyword_retriever=c['keyword_retriever'],
            response_cache=ResponseCache()
        )
        
        self.components['rag_orchestrator'] = rag_orchestrator
        logger.info("RAG orchestrator built.")