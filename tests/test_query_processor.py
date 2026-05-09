import pytest
from src.core.query_processor import QueryProcessor

@pytest.fixture
def processor():
    """Provides a QueryProcessor instance."""
    return QueryProcessor()

def test_normalize_basic(processor):
    """Tests basic normalization of a query."""
    query = "  What is  the capital of France?   "
    expected = "what is the capital of france?"
    assert processor.normalize(query) == expected

def test_normalize_empty_string(processor):
    """Tests normalization of an empty string."""
    assert processor.normalize("") == ""

def test_normalize_already_normalized(processor):
    """Tests a query that is already normalized."""
    query = "this is a test."
    assert processor.normalize(query) == "this is a test."

def test_clean_for_keywords_basic(processor):
    """Tests basic keyword cleaning."""
    query = "What is the best way to learn about RAG systems?"
    expected = ["what", "best", "way", "learn", "rag", "systems"]
    assert processor.clean_for_keywords(query) == expected

def test_clean_for_keywords_with_punctuation(processor):
    """Tests keyword cleaning with various punctuation."""
    query = "Auto-Adaptive RAG, what's the big deal?"
    expected = ["autoadaptive", "rag", "whats", "big", "deal"]
    assert processor.clean_for_keywords(query) == expected

def test_clean_for_keywords_empty_string(processor):
    """Tests keyword cleaning of an empty string."""
    assert processor.clean_for_keywords("") == []

def test_clean_for_keywords_only_stopwords(processor):
    """Tests a query that only contains stop words."""
    query = "a the an of"
    assert processor.clean_for_keywords(query) == []