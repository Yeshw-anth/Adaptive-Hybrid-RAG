import pytest
import os
import networkx as nx
from unittest.mock import patch, MagicMock, ANY
from src.core.graph.graph_store import NetworkxGraphStore

@pytest.fixture
def temp_graph_path(tmp_path):
    """Provides a temporary path for the graph file."""
    return os.path.join(tmp_path, "test_graph.graphml")

def test_initialization_new_graph(temp_graph_path):
    """Tests that a new graph is created if one doesn't exist."""
    store = NetworkxGraphStore(graph_path=temp_graph_path)
    assert store.graph is not None
    assert isinstance(store.graph, nx.DiGraph)
    assert store.graph.number_of_nodes() == 0

def test_initialization_load_existing_graph(temp_graph_path):
    """Tests that an existing graph is loaded correctly."""
    # Arrange: Create and save a dummy graph
    g = nx.DiGraph()
    g.add_node("test_node", type="document")
    nx.write_graphml(g, temp_graph_path)

    # Act
    store = NetworkxGraphStore(graph_path=temp_graph_path)

    # Assert
    assert store.graph.number_of_nodes() == 1
    assert "test_node" in store.graph.nodes

@patch('networkx.read_graphml', side_effect=Exception("Corrupted file"))
def test_initialization_corrupted_graph(mock_read_graphml, temp_graph_path):
    """Tests that a new graph is created if the existing one is corrupted."""
    # Arrange: Create a dummy file to simulate existence
    with open(temp_graph_path, "w") as f:
        f.write("corrupted data")

    # Act
    store = NetworkxGraphStore(graph_path=temp_graph_path)

    # Assert
    assert store.graph is not None
    assert store.graph.number_of_nodes() == 0
    mock_read_graphml.assert_called_once_with(temp_graph_path)

def test_add_and_has_source_document(temp_graph_path):
    """Tests adding and checking for a source document."""
    # Arrange
    store = NetworkxGraphStore(graph_path=temp_graph_path)
    filename = "test_doc.pdf"
    file_hash = "some_hash_value"

    # Act
    assert not store.has_source_document(filename, file_hash)
    store.add_source_document(filename, file_hash)

    # Assert
    assert store.has_source_document(filename, file_hash)
    node_attrs = store.graph.nodes[filename]
    assert node_attrs['type'] == 'source_document'
    assert node_attrs['file_hash'] == file_hash

def test_add_node_and_edge(temp_graph_path):
    """Tests adding nodes and edges to the graph."""
    # Arrange
    store = NetworkxGraphStore(graph_path=temp_graph_path)

    # Act
    store.add_node("subject_node")
    store.add_node("object_node", color="blue")
    store.add_edge("subject_node", "object_node", label="connects_to")

    # Assert
    assert "subject_node" in store.graph.nodes
    assert store.graph.nodes["object_node"]["color"] == "blue"
    assert store.graph.has_edge("subject_node", "object_node")
    edge_data = store.graph.get_edge_data("subject_node", "object_node")
    assert edge_data["label"] == "connects_to"

@patch('src.core.graph.graph_store.NetworkxGraphStore._save_graph_sync', return_value=True)
def test_save_graph_submission(mock_save_sync, temp_graph_path):
    """Tests that the save operation is correctly submitted to the executor."""
    # Arrange
    store = NetworkxGraphStore(graph_path=temp_graph_path)

    # Act
    save_future = store.save_graph()
    save_future.result() # Wait for the future to complete

    # Assert
    mock_save_sync.assert_called_once()
    assert store._save_future is not None
    assert save_future.result() is True