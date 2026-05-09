import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.decision.confidence_engine import ConfidenceEngine, GroundednessCheck
from llama_index.core.schema import TextNode, NodeWithScore

@pytest.fixture
def mock_llm_wrapper():
    """Provides a mock LLM wrapper."""
    return AsyncMock()

@pytest.fixture
def confidence_engine(mock_llm_wrapper):
    """Provides a ConfidenceEngine instance with a mocked LLM wrapper."""
    return ConfidenceEngine(mock_llm_wrapper)

def test_calculate_context_confidence_no_docs(confidence_engine):
    """
    Tests that context confidence is 0 when no documents are provided.
    """
    result = confidence_engine.calculate_context_confidence({}, [])
    assert "confidence_score" in result
    assert result["confidence_score"] == 0

def test_calculate_context_confidence_with_retrieved_docs(confidence_engine):
    """
    Tests context confidence calculation with only retrieved documents.
    """
    retrieved_docs = [
        NodeWithScore(node=TextNode(text="doc1"), score=0.9),
        NodeWithScore(node=TextNode(text="doc2"), score=0.8),
    ]
    result = confidence_engine.calculate_context_confidence({}, retrieved_docs)
    assert "context_score" in result
    assert result["context_score"] > 0
    assert result["details"]["reranker_score_avg"] == 0

def test_calculate_context_confidence_with_reranked_docs(confidence_engine):
    """
    Tests context confidence calculation with reranked documents.
    """
    retrieved_docs = [
        NodeWithScore(node=TextNode(text="doc1"), score=0.7),
        NodeWithScore(node=TextNode(text="doc2"), score=0.6),
    ]
    reranked_docs = [
        NodeWithScore(node=TextNode(text="doc1"), score=4.5), # High positive score
        NodeWithScore(node=TextNode(text="doc2"), score=-1.2), # Negative score
    ]
    result = confidence_engine.calculate_context_confidence({}, retrieved_docs, reranked_docs)
    assert "context_score" in result
    assert result["context_score"] > 0
    assert result["details"]["reranker_score_avg"] > 0
    assert result["details"]["relevance_score_avg"] > 0

def test_decide_action_from_context(confidence_engine):
    """
    Tests the logic for deciding an action based on context confidence.
    """
    assert confidence_engine.decide_action_from_context({"context_score": 0.9}) == "generate"
    assert confidence_engine.decide_action_from_context({"context_score": 0.6}) == "expand"
    assert confidence_engine.decide_action_from_context({"context_score": 0.4}) == "expand"
    assert confidence_engine.decide_action_from_context({"context_score": 0.6}, is_post_expansion=True) == "generate"


@pytest.mark.asyncio
@patch('src.core.decision.confidence_engine.settings')
async def test_verify_groundedness_happy_path(mock_settings, confidence_engine, mock_llm_wrapper):
    """
    Tests the happy path for the groundedness verification.
    """
    # Arrange
    mock_settings.LARGE_LLM_MODEL = "test-large-model"
    answer = "The sky is blue."
    context_docs = [NodeWithScore(node=TextNode(text="The sky is indeed blue."))]
    mock_response = {
        "content": '{"is_grounded": true, "unsupported_claims": [], "groundedness_score": 1.0}',
        "token_usage": {"input_tokens": 10, "output_tokens": 5}
    }
    mock_llm_wrapper.generate_from_prompt.return_value = mock_response

    # Act
    result = await confidence_engine.verify_groundedness(answer, context_docs)

    # Assert
    mock_llm_wrapper.generate_from_prompt.assert_called_once()
    assert result.is_grounded is True
    assert result.groundedness_score == 1.0

@pytest.mark.asyncio
async def test_verify_groundedness_llm_error(confidence_engine, mock_llm_wrapper):
    """
    Tests that groundedness verification fails gracefully on LLM error.
    """
    # Arrange
    answer = "The sky is blue."
    context_docs = [NodeWithScore(node=TextNode(text="The sky is indeed blue."))]
    mock_llm_wrapper.generate_from_prompt.side_effect = Exception("LLM failed")

    # Act
    result = await confidence_engine.verify_groundedness(answer, context_docs)

    # Assert
    assert result.is_grounded is False
    assert result.groundedness_score == 0.0
    assert result.unsupported_claims == [answer]

def test_get_final_confidence(confidence_engine):
    """
    Tests the calculation of the final confidence score.
    """
    context_confidence = {"context_score": 0.8, "details": {}}
    groundedness_check = GroundednessCheck(is_grounded=True, unsupported_claims=[], groundedness_score=0.9)
    query_metadata = {"complexity": "low"}

    result = confidence_engine.get_final_confidence(context_confidence, groundedness_check, query_metadata)

    assert "confidence_score" in result
    assert result["confidence_level"] == "high"
    # Expected: (0.3 * 0.8) + (0.7 * 0.9) = 0.24 + 0.63 = 0.87
    assert result["confidence_score"] == pytest.approx(0.87)