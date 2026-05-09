import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.pipelines.fast_pipeline import FastPipeline
from src.data.schemas import QueryMetadata, Strategy
from llama_index.core.schema import TextNode, NodeWithScore

@pytest.fixture
def mock_dependencies():
    """Provides a dictionary of mocked dependencies for the FastPipeline."""
    # The retrieve method on the graph_retriever is async and needs to be awaited.
    # We ensure the mock is an AsyncMock to correctly simulate this behavior.
    graph_retriever_mock = AsyncMock()
    graph_retriever_mock.retrieve = AsyncMock()

    return {
        "retriever": MagicMock(),
        "graph_retriever": graph_retriever_mock,
        "llm_client": AsyncMock(),
        "confidence_engine": MagicMock(),
    }

@pytest.fixture
def fast_pipeline(mock_dependencies):
    """Provides a FastPipeline instance with mocked dependencies."""
    return FastPipeline(**mock_dependencies)

@pytest.mark.asyncio
async def test_execute_vector_retrieval_happy_path(fast_pipeline, mock_dependencies):
    """
    Tests the happy path for the execute method with vector retrieval.
    """
    # Arrange
    query = "test query"
    query_analysis = {
        "metadata": QueryMetadata(intent="fact-seeking", complexity="low", keywords=["test"], expected_answer_format="single_value", query_type="simple", normalized_query="test query", keyword_tokens=["test"]),
        "strategy": Strategy(pipeline="fast", model="test_model", retrieval_strategy="vector", use_reranker=False, top_k=3)
    }
    mock_nodes = [NodeWithScore(node=TextNode(text="test context", id_="1"), score=0.9)]
    mock_dependencies["retriever"].retrieve.return_value = mock_nodes
    mock_dependencies["confidence_engine"].calculate_context_confidence.return_value = {"context_score": 0.9}
    mock_dependencies["confidence_engine"].decide_action_from_context.return_value = "proceed"
    mock_dependencies["llm_client"].generate_structured_response.return_value = "test answer"

    # Act
    result = await fast_pipeline.execute(query, query_analysis)

    # Assert
    mock_dependencies["retriever"].retrieve.assert_called_once_with(query, top_k=3)
    mock_dependencies["llm_client"].generate_structured_response.assert_called_once()
    assert result["answer"] == "test answer"
    assert result["pipeline"] == "fast"

@pytest.mark.asyncio
async def test_execute_graph_retrieval_happy_path(fast_pipeline, mock_dependencies):
    """
    Tests the happy path for the execute method with graph retrieval.
    """
    # Arrange
    query = "test query"
    query_analysis = {
        "metadata": QueryMetadata(intent="fact-seeking", complexity="low", keywords=["test"], expected_answer_format="single_value", query_type="simple", normalized_query="test query", keyword_tokens=["test"]),
        "strategy": Strategy(pipeline="fast", model="test_model", retrieval_strategy="hybrid", use_reranker=False, top_k=5)
    }
    mock_dependencies["graph_retriever"].retrieve.return_value = "graph context"
    mock_dependencies["confidence_engine"].calculate_context_confidence.return_value = {"context_score": 0.9}
    mock_dependencies["confidence_engine"].decide_action_from_context.return_value = "proceed"
    mock_dependencies["llm_client"].generate_structured_response.return_value = "test answer"

    # Act
    result = await fast_pipeline.execute(query, query_analysis)

    # Assert
    mock_dependencies["graph_retriever"].retrieve.assert_called_once_with(query)
    # Since the graph retriever was successful, the vector retriever should not be called.
    mock_dependencies["retriever"].retrieve.assert_not_called()
    mock_dependencies["llm_client"].generate_structured_response.assert_called_once()
    assert result["answer"] == "test answer"

@pytest.mark.asyncio
async def test_execute_abstain(fast_pipeline, mock_dependencies):
    """
    Tests that the pipeline abstains when confidence is low.
    """
    # Arrange
    query = "test query"
    query_analysis = {
        "metadata": QueryMetadata(intent="fact-seeking", complexity="low", keywords=["test"], expected_answer_format="single_value", query_type="simple", normalized_query="test query", keyword_tokens=["test"]),
        "strategy": Strategy(pipeline="fast", model="test_model", retrieval_strategy="vector", use_reranker=False, top_k=3)
    }
    mock_nodes = [NodeWithScore(node=TextNode(text="test context", id_="1"), score=0.1)]
    mock_dependencies["retriever"].retrieve.return_value = mock_nodes
    mock_dependencies["confidence_engine"].calculate_context_confidence.return_value = {"context_score": 0.1}
    mock_dependencies["confidence_engine"].decide_action_from_context.return_value = "abstain"

    # Act
    result = await fast_pipeline.execute(query, query_analysis)

    # Assert
    mock_dependencies["retriever"].retrieve.assert_called_once()
    mock_dependencies["llm_client"].generate_structured_response.assert_not_called()
    assert result["answer"] == "Could not find any relevant information."

@pytest.mark.asyncio
async def test_execute_no_context(fast_pipeline, mock_dependencies):
    """
    Tests the case where no context is retrieved.
    """
    # Arrange
    query = "test query"
    query_analysis = {
        "metadata": QueryMetadata(intent="fact-seeking", complexity="low", keywords=["test"], expected_answer_format="single_value", query_type="simple", normalized_query="test query", keyword_tokens=["test"]),
        "strategy": Strategy(pipeline="fast", model="test_model", retrieval_strategy="vector", use_reranker=False, top_k=3)
    }
    mock_dependencies["retriever"].retrieve.return_value = []

    # Act
    result = await fast_pipeline.execute(query, query_analysis)

    # Assert
    mock_dependencies["retriever"].retrieve.assert_called_once()
    mock_dependencies["llm_client"].generate_structured_response.assert_not_called()
    assert result["answer"] == "Could not find any relevant information."