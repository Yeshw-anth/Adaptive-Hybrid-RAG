import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.strategy.query_analyzer import QueryAnalyzer, LLMAnalysisResponse
from src.data.schemas import QueryMetadata
import json

@pytest.fixture
def mock_llm_wrapper():
    """Provides a mock LLM wrapper."""
    return AsyncMock()

@pytest.fixture
def query_analyzer(mock_llm_wrapper):
    """Provides a QueryAnalyzer instance with a mocked LLM wrapper."""
    return QueryAnalyzer(mock_llm_wrapper, default_llm_model="test_model")

@pytest.mark.asyncio
async def test_analyze_happy_path(query_analyzer, mock_llm_wrapper):
    """
    Tests the happy path for the analyze method.
    """
    # Arrange
    query = "test query"
    keyword_tokens = ["test", "query"]
    mock_response_content = {
        "intent": "fact-seeking",
        "complexity": "low",
        "keywords": ["test"],
        "expected_answer_format": "single_value",
        "query_type": "simple",
        "retrieval_strategy": "vector"
    }
    mock_response = {
        "content": json.dumps(mock_response_content),
        "token_usage": {"input_tokens": 10, "output_tokens": 5}
    }
    mock_llm_wrapper.generate_with_system_prompt.return_value = mock_response

    # Act
    result = await query_analyzer.analyze(query, keyword_tokens)

    # Assert
    mock_llm_wrapper.generate_with_system_prompt.assert_called_once()
    assert isinstance(result, QueryMetadata)
    assert result.intent == "fact-seeking"
    assert result.complexity == "low"

@pytest.mark.asyncio
async def test_analyze_llm_failure_and_retry(query_analyzer, mock_llm_wrapper):
    """
    Tests that the analyzer retries on LLM failure and succeeds on the second attempt.
    """
    # Arrange
    query = "test query"
    keyword_tokens = ["test", "query"]
    mock_valid_response_content = {
        "intent": "summary",
        "complexity": "high",
        "keywords": ["summary"],
        "expected_answer_format": "explanation",
        "query_type": "complex",
        "retrieval_strategy": "hybrid"
    }
    mock_llm_wrapper.generate_with_system_prompt.side_effect = [
        Exception("LLM call failed"),
        {
            "content": json.dumps(mock_valid_response_content),
            "token_usage": {"input_tokens": 10, "output_tokens": 5}
        }
    ]

    # Act
    result = await query_analyzer.analyze(query, keyword_tokens)

    # Assert
    assert mock_llm_wrapper.generate_with_system_prompt.call_count == 2
    assert result.intent == "summary"

@pytest.mark.asyncio
async def test_analyze_fallback_on_persistent_failure(query_analyzer, mock_llm_wrapper):
    """
    Tests that the analyzer falls back to default metadata after multiple failures.
    """
    # Arrange
    query = "test query"
    keyword_tokens = ["test", "query"]
    mock_llm_wrapper.generate_with_system_prompt.side_effect = Exception("Persistent failure")
    # The fixture-provided analyzer is used, but we need to reset the mock's call count
    # as it's shared across parameterizations of a test or other non-obvious sharing.
    mock_llm_wrapper.generate_with_system_prompt.reset_mock()
    query_analyzer.max_retries = 1 # Limit retries for the test

    # Act
    result = await query_analyzer.analyze(query, keyword_tokens)

    # Assert
    assert mock_llm_wrapper.generate_with_system_prompt.call_count == 2
    assert result.intent == "fact-seeking" # Default value
    assert result.complexity == "high" # Default value

def test_extract_and_parse_json(query_analyzer):
    """
    Tests the JSON extraction method with various formats.
    """
    # Basic JSON
    assert query_analyzer._extract_and_parse_json('{"key": "value"}') == {"key": "value"}
    # JSON with markdown fence
    assert query_analyzer._extract_and_parse_json('```json\n{"key": "value"}\n```') == {"key": "value"}
    # JSON with text before and after
    assert query_analyzer._extract_and_parse_json('Some text\n{"key": "value"}\nSome more text') == {"key": "value"}
    # Test with single quotes fix
    assert query_analyzer._extract_and_parse_json("{'key': 'value'}") == {"key": "value"}
    # Test with trailing comma fix
    assert query_analyzer._extract_and_parse_json('{"key": "value",}') == {"key": "value"}