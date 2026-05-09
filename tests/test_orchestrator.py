import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.orchestrator import RAGOrchestrator
from src.data.schemas import QueryMetadata, Strategy

@pytest.fixture
def mock_dependencies():
    """Provides a dictionary of mocked dependencies for the RAGOrchestrator."""
    return {
        "query_analyzer": AsyncMock(),
        "strategy_router": MagicMock(),
        "query_expander": AsyncMock(),
        "pipelines": {"accurate": AsyncMock(), "fast": AsyncMock()},
        "confidence_engine": AsyncMock(),
        "llm_client": AsyncMock(),
        "retriever": MagicMock(),
        "ingestion_pipeline": AsyncMock(),
        "vector_index": MagicMock(),
        "hybrid_retriever": MagicMock(),
        "keyword_retriever": MagicMock(),
        "response_cache": MagicMock(),
    }

@pytest.fixture
def orchestrator(mock_dependencies):
    """Provides a RAGOrchestrator instance with mocked dependencies."""
    return RAGOrchestrator(**mock_dependencies)

@pytest.mark.asyncio
async def test_query_happy_path(orchestrator, mock_dependencies):
    """
    Tests the main query method's happy path.
    """
    # Arrange
    query = "test query"
    mock_metadata = QueryMetadata(intent="fact-seeking", complexity="low", keywords=["test"], expected_answer_format="explanation", query_type="simple", normalized_query="test query", keyword_tokens=["test"])
    mock_strategy = Strategy(pipeline="accurate", model="test_model", retrieval_strategy="vector", use_reranker=False, top_k=5)
    mock_pipeline_result = {"answer": "test answer"}

    mock_dependencies["query_analyzer"].analyze.return_value = mock_metadata
    mock_dependencies["strategy_router"].select_strategy.return_value = mock_strategy
    mock_dependencies["pipelines"]["accurate"].execute.return_value = mock_pipeline_result

    # Act
    result = await orchestrator.query(query)

    # Assert
    mock_dependencies["query_analyzer"].analyze.assert_called_once()
    mock_dependencies["strategy_router"].select_strategy.assert_called_once_with(mock_metadata, model_override=None)
    mock_dependencies["pipelines"]["accurate"].execute.assert_called_once()
    assert result["answer"] == "test answer"
    assert result["strategy"] == mock_strategy

@pytest.mark.asyncio
async def test_query_analysis_failure_fallback(orchestrator, mock_dependencies):
    """
    Tests that the orchestrator falls back to a default strategy if query analysis fails.
    """
    # Arrange
    query = "test query"
    mock_dependencies["query_analyzer"].analyze.side_effect = Exception("Analysis failed")
    mock_strategy = Strategy(pipeline="accurate", model="default_model", retrieval_strategy="vector", use_reranker=False, top_k=5)
    mock_dependencies["strategy_router"].select_strategy.return_value = mock_strategy
    mock_pipeline_result = {"answer": "fallback answer"}
    mock_dependencies["pipelines"]["accurate"].execute.return_value = mock_pipeline_result

    # Act
    result = await orchestrator.query(query)

    # Assert
    mock_dependencies["query_analyzer"].analyze.assert_called_once()
    mock_dependencies["strategy_router"].select_strategy.assert_called_once()
    mock_dependencies["pipelines"]["accurate"].execute.assert_called_once()
    assert result["answer"] == "fallback answer"

@pytest.mark.asyncio
async def test_ingest_and_process(orchestrator, mock_dependencies):
    """
    Tests the ingestion process.
    """
    # Arrange
    file_path = "test.pdf"
    mock_nodes = [MagicMock()]
    mock_dependencies["ingestion_pipeline"].ingest_file.return_value = mock_nodes
    
    # Mock the docstore and its values
    mock_docstore = MagicMock()
    mock_docstore.docs.values.return_value = mock_nodes
    mock_dependencies["vector_index"].docstore = mock_docstore
    
    # Mock the storage_context for the persist call
    mock_storage_context = MagicMock()
    mock_dependencies["vector_index"].storage_context = mock_storage_context

    # Act
    result = await orchestrator.ingest_and_process(file_path)

    # Assert
    mock_dependencies["ingestion_pipeline"].ingest_file.assert_called_once_with(file_path)
    mock_dependencies["vector_index"].insert_nodes.assert_called_once_with(mock_nodes)
    mock_storage_context.persist.assert_called_once()
    mock_dependencies["keyword_retriever"].update_corpus.assert_called_once()
    mock_dependencies["hybrid_retriever"].update_corpus.assert_called_once()
    assert len(result) == len(mock_nodes)