import pytest
from unittest.mock import MagicMock, AsyncMock

from src.core.pipelines.accurate_pipeline import AccuratePipeline
from src.data.schemas import Strategy, QueryMetadata
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.fixture
def mock_node_with_score():
    """Creates a mock NodeWithScore object."""
    node = TextNode(text="This is a test document.", id_="test_node_1")
    node.metadata = {"file_path": "/fake/path/doc.txt"}
    return NodeWithScore(node=node, score=0.9)


@pytest.mark.asyncio
async def test_process_query_with_vector_strategy(mocker, mock_node_with_score):
    """
    Tests that the AccuratePipeline is used when the retrieval strategy is 'vector'.
    """
    # Mocks
    mock_retriever = MagicMock()
    mock_hybrid_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_llm_client = MagicMock()
    mock_query_expander = MagicMock()
    mock_confidence_engine = MagicMock()
    mock_hybrid_graph_retriever = MagicMock()

    # Mock the retriever's response
    mock_retriever.retrieve.return_value = [mock_node_with_score]
    
    # Mock the reranker's response
    mock_reranker.rerank_nodes.return_value = [mock_node_with_score]

    # Mock the confidence engine's decision
    mock_confidence_engine.calculate_context_confidence.return_value = {"final_score": 0.9}
    mock_confidence_engine.decide_action_from_context.return_value = "generate"

    # Mock the LLM's response
    mock_llm_client.generate_structured_response = AsyncMock(return_value={"content": "vector answer"})

    pipeline = AccuratePipeline(
        retriever=mock_retriever,
        hybrid_retriever=mock_hybrid_retriever,
        reranker=mock_reranker,
        llm_client=mock_llm_client,
        query_expander=mock_query_expander,
        confidence_engine=mock_confidence_engine,
        hybrid_graph_retriever=mock_hybrid_graph_retriever
    )
    
    query_analysis = {
        'strategy': Strategy(pipeline="accurate", retrieval_strategy='vector', use_reranker=True, use_parent_child=False, model='phi3', top_k=5, reranker_top_n=3),
        'metadata': QueryMetadata(normalized_query="test query", keyword_tokens=["test", "query"], query_type="simple", intent="fact-seeking")
    }

    # Execution
    result = await pipeline.execute("test query", query_analysis)
    
    # Assertions
    mock_retriever.retrieve.assert_called_once()
    mock_reranker.rerank_nodes.assert_called_once()
    mock_llm_client.generate_structured_response.assert_called_once()
    assert result["answer"] == "vector answer"