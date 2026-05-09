import pytest
from unittest.mock import patch, MagicMock
from src.core.system_builder import SystemBuilder
from src.config.settings import Settings
import src.core.system_builder as system_builder_module

@pytest.fixture
def mock_settings():
    """Provides mock settings for the SystemBuilder."""
    settings = MagicMock(spec=Settings)
    settings.EMBED_MODEL_NAME = "test-embedding-model"
    settings.EMBEDDING_DIM = 384
    settings.VECTOR_STORE_PATH = MagicMock()
    settings.VECTOR_STORE_PATH.exists.return_value = False
    settings.VECTOR_STORE_PATH.iterdir.return_value = []
    settings.DEFAULT_LLM_MODEL = "test-default-model"
    return settings

@patch('src.chunking.semantic_segmenter.SentenceTransformer')
@patch('src.core.system_builder.StorageContext')
@patch('src.core.system_builder.VectorStoreIndex')
@patch('src.core.system_builder.faiss')
@patch('src.core.system_builder.OllamaClient')
@patch('src.core.system_builder.Embedder')
def test_system_builder_initialization(
    mock_embedder, mock_ollama_client, mock_faiss,
    mock_vector_store_index, mock_storage_context, mock_sentence_transformer, mock_settings
):
    """
    Tests that the SystemBuilder initializes all its components correctly.
    """
    # Arrange
    builder = SystemBuilder(mock_settings)

    # Act
    builder.build_all()

    # Assert
    assert "llm_client" in builder.components
    assert "embedder" in builder.components
    assert "vector_index" in builder.components
    assert "rag_orchestrator" in builder.components
    mock_embedder.assert_called_once_with(model_name="test-embedding-model")
    mock_ollama_client.assert_called_once()