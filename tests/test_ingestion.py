import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.core.ingestion import IngestionPipeline, get_file_hash
from llama_index.core.schema import TextNode

@pytest.fixture
def mock_dependencies():
    """Provides a dictionary of mocked dependencies for the IngestionPipeline."""
    return {
        "chunking_engine": AsyncMock(),
        "graph_builder": AsyncMock(),
        "llm_client": MagicMock(),
        "graph_store": MagicMock(),
        "vector_index": MagicMock(),
    }

@pytest.fixture
def ingestion_pipeline(mock_dependencies):
    """Provides an instance of the IngestionPipeline with mocked dependencies."""
    pipeline = IngestionPipeline(
        chunking_engine=mock_dependencies["chunking_engine"],
        graph_builder=mock_dependencies["graph_builder"],
        llm_client=mock_dependencies["llm_client"],
        graph_store=mock_dependencies["graph_store"],
        vector_index=mock_dependencies["vector_index"],
    )
    # Mock the internal method to avoid actual graph building in this test
    pipeline._build_graph_from_nodes = AsyncMock()
    return pipeline


@pytest.mark.asyncio
@patch('src.core.ingestion.get_file_hash', return_value="dummy_hash")
@patch('src.core.ingestion.IngestionRouter.get_loader')
async def test_ingest_file_already_exists(mock_get_loader, mock_get_file_hash, ingestion_pipeline, mock_dependencies):
    """
    Tests that no action is taken when a file is already fully ingested.
    """
    # Arrange
    file_path = "test.pdf"
    mock_dependencies['graph_store'].has_source_document.return_value = True
    mock_nodes = [TextNode(text="test node", metadata={"file_path": file_path})]
    mock_dependencies['vector_index'].docstore.docs.values.return_value = mock_nodes

    # Act
    await ingestion_pipeline.ingest_file(file_path)

    # Assert
    mock_dependencies['graph_store'].has_source_document.assert_called_once_with("test.pdf", "dummy_hash")
    mock_get_loader.assert_not_called()
    mock_dependencies['chunking_engine'].chunk_document.assert_not_called()
    mock_dependencies['vector_index'].insert_nodes.assert_not_called()
    assert not ingestion_pipeline._build_graph_from_nodes.called
    mock_dependencies['graph_store'].add_source_document.assert_not_called()
    mock_dependencies['graph_store'].save_graph.assert_not_called()