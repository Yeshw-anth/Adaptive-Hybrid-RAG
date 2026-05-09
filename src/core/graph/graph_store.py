import json
import os
import networkx as nx
from typing import Protocol, Any
from src.core.logging_config import logger
from src.config.settings import settings
import time
from concurrent.futures import ThreadPoolExecutor, Future


class GraphStore(Protocol):
    """
    A protocol defining the interface for a graph store, which manages the
    knowledge graph's nodes, edges, and persistence.
    """
    def has_source_document(self, filename: str, file_hash: str) -> bool:
        """
        Checks if a source document node with a specific hash already exists.
        """
        ...

    def add_source_document(self, filename: str, file_hash: str):
        """
        Adds or updates a source document node with its file hash.
        """
        ...

    def add_node(self, node_id: str, **kwargs: Any):
        """Adds a node to the graph if it doesn't exist, or updates it."""
        ...

    def add_edge(self, source_id: str, target_id: str, **kwargs: Any):
        """Adds an edge to the graph if it doesn't exist."""
        ...

    def save_graph(self) -> "Future[Any]":
        """Saves the current state of the graph to its persistent storage."""
        ...

    def shutdown(self):
        """Performs any necessary cleanup or saving on application shutdown."""
        ...


class NetworkxGraphStore:
    """
    Manages the loading and saving of the knowledge graph.
    """
    def __init__(self, graph_path: str):
        if not graph_path:
            raise ValueError("A graph_path must be provided to initialize the GraphStore.")

        self.graph_path = graph_path
        # Ensure the directory for the graph file exists
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        self.graph = self._load_graph()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='GraphSaveThread')
        self._save_future: Future | None = None
        logger.info(f"GraphStore initialized for path: {self.graph_path}")

    def _load_graph(self):
        """
        Loads the graph from the primary GraphML file if it exists and is valid.
        Otherwise, creates a new graph.
        """
        if os.path.exists(self.graph_path) and os.path.getsize(self.graph_path) > 0:
            try:
                graph = nx.read_graphml(self.graph_path)
                logger.info(f"Successfully loaded existing graph from {self.graph_path} "
                            f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges).")
                return graph
            except Exception as e:
                logger.error(f"Error loading graph from {self.graph_path}: {e}. "
                             f"The file might be corrupted. Creating a new graph.")
                return nx.DiGraph()
        else:
            logger.info(f"No valid graph found at {self.graph_path}. Creating a new, empty graph.")
            return nx.DiGraph()



    def _save_graph_sync(self) -> bool:
        """Synchronously saves the graph to a file with atomic replacement."""
        # If a save is already in progress, wait for it to complete first
        if self._save_future and self._save_future.running():
            logger.info("A graph save is already in progress. Waiting for it to complete.")
            self._save_future.result() # Wait for the future to finish

        temp_graph_path = self.graph_path.with_suffix('.graphml.tmp')
        try:
            graphml_lines = nx.generate_graphml(self.graph, encoding='utf-8', prettyprint=True)
            with open(temp_graph_path, 'w', encoding='utf-8') as f:
                for line in graphml_lines:
                    f.write(line)

            os.replace(temp_graph_path, self.graph_path)

            logger.info(f"Successfully and atomically saved graph to {self.graph_path}")
            return True
        except Exception as e:
            logger.error(f"Primary save to GraphML failed: {e}", exc_info=True)
            # Fallback mechanism remains the same
            return False
        finally:
            if temp_graph_path.exists():
                temp_graph_path.unlink()

    def save_graph(self) -> Future:
        """
        Submits the graph saving operation to a background thread and returns a Future.
        This method returns immediately.
        """
        logger.info("Submitting graph save task to background thread.")
        self._save_future = self._executor.submit(self._save_graph_sync)
        return self._save_future

    def shutdown(self):
        """Saves the graph on shutdown, waiting for any pending save to complete."""
        logger.info("GraphStore shutdown initiated.")
        if self._save_future and self._save_future.running():
            logger.info("Waiting for the background graph save to complete...")
            self._save_future.result() # Block and wait for the future to finish
            logger.info("Background save completed.")

        # Perform a final synchronous save to capture any last-minute changes
        logger.info("Performing final synchronous graph save on shutdown.")
        self._save_graph_sync()

        self._executor.shutdown(wait=True)
        logger.info("GraphStore shutdown complete.")

    def has_source_document(self, filename: str, file_hash: str) -> bool:
        """
        Checks if a source document node with a specific hash already exists in the graph.
        """
        node = self.graph.nodes.get(filename)
        if not node:
            return False

        is_source_doc = node.get("type") == "source_document"
        has_matching_hash = node.get("file_hash") == file_hash
        
        return is_source_doc and has_matching_hash

    def add_source_document(self, filename: str, file_hash: str):
        """
        Adds or updates a source document node in the graph, marking it as processed
        with its specific file hash.
        """
        self.add_node(
            filename,
            type="source_document",
            timestamp=time.time(),
            file_hash=file_hash
        )
        logger.info(f"Added/updated source document node for '{filename}' with hash '{file_hash[:8]}...'.")

    def add_node(self, node_id, **kwargs):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **kwargs)
            logger.debug(f"Added node {node_id} to the graph.")
        else:
            # Update existing node attributes
            for key, value in kwargs.items():
                self.graph.nodes[node_id][key] = value
            logger.debug(f"Updated attributes for existing node {node_id}.")

    def add_edge(self, source_id, target_id, **kwargs):
        if not self.graph.has_edge(source_id, target_id):
            self.graph.add_edge(source_id, target_id, **kwargs)
            logger.debug(f"Added edge from {source_id} to {target_id}.")

    def get_node(self, node_id):
        return self.graph.nodes.get(node_id)

    def get_all_nodes(self):
        return list(self.graph.nodes(data=True))

    def get_all_edges(self):
        return list(self.graph.edges(data=True))

    def query_graph(self, query: str):
        # This is a placeholder for a more sophisticated graph query mechanism.
        # For now, it just returns nodes that contain the query string in their attributes.
        logger.info(f"Executing simple query for: '{query}'")
        results = []
        for node, data in self.graph.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, str) and query.lower() in value.lower():
                    results.append((node, data))
                    break
        logger.info(f"Query found {len(results)} matching nodes.")
        return results


def get_graph_store():
    """
    Factory function to get the appropriate graph store based on settings.
    """
    store_type = settings.GRAPH_STORE_TYPE.lower()
    logger.info(f"Initializing graph store of type: {store_type}")
    if store_type == "neo4j":
        from .neo4j_graph_store import Neo4jGraphStore
        return Neo4jGraphStore()
    elif store_type == "networkx":
        return NetworkxGraphStore(graph_path=settings.GRAPH_FILE_PATH)
    else:
        raise ValueError(f"Unknown GRAPH_STORE_TYPE: {store_type}")