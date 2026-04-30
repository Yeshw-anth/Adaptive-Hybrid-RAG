import logging
import json
import os
import networkx as nx
from src.core.logging_config import logger
from src.config.settings import settings

class NetworkxGraphStore:
    """
    Manages the loading and saving of the knowledge graph.
    """
    def __init__(self, graph_path: str = settings.GRAPH_PATH):
        self.graph_path = graph_path
        self.graph = self._load_graph()
        logger.info(f"GraphStore initialized with path: {self.graph_path}")

    def _load_graph(self):
        temp_path = self.graph_path.with_suffix('.temp.json')

        # Priority 1: Recover from a temporary JSON file from a failed previous run.
        if temp_path.exists():
            logger.warning(f"Found temporary graph file at {temp_path}. Attempting recovery.")
            try:
                with open(temp_path, 'r') as f:
                    data = json.load(f)
                graph = nx.node_link_graph(data)
                logger.info(f"Successfully loaded graph from temporary file. Contains {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
                
                # Immediately attempt to save the recovered graph to the primary .graphml file.
                # This is the "save in graph ml from json file" step.
                self.graph = graph # Temporarily set self.graph to the recovered graph for saving
                if self.save_graph():
                    logger.info("Recovery successful: Graph has been saved to primary GraphML format.")
                    os.remove(temp_path)
                    logger.info(f"Removed temporary file: {temp_path}")
                else:
                    logger.error("Recovery failed: Could not save graph to primary format. The temporary file has been kept.")
                return graph
            except Exception as e:
                logger.error(f"Error recovering from temporary file {temp_path}: {e}. Proceeding to normal load.")

        # Priority 2: If no temp file, load from the primary GraphML file as normal.
        if self.graph_path.exists():
            try:
                graph = nx.read_graphml(self.graph_path)
                logger.info(f"Loaded existing graph from {self.graph_path} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
                return graph
            except Exception as e:
                logger.error(f"Error loading graph from {self.graph_path}: {e}. Creating a new graph.")
                return nx.DiGraph()
        
        # Priority 3: If no files exist, create a new graph.
        else:
            logger.info(f"No graph found at {self.graph_path}. Creating a new graph.")
            return nx.DiGraph()



    def save_graph(self) -> bool:
        try:
            nx.write_graphml(self.graph, self.graph_path)
            logger.info(f"Successfully saved graph to {self.graph_path}")
            return True
        except Exception as e:
            logger.error(f"Primary save to GraphML failed: {e}")
            logger.warning("Attempting to save a temporary JSON fallback...")
            try:
                temp_path = self.graph_path.with_suffix('.temp.json')
                data = nx.node_link_data(self.graph)
                with open(temp_path, 'w') as f:
                    json.dump(data, f)
                logger.info(f"Successfully saved JSON fallback to {temp_path}")
            except Exception as fallback_e:
                logger.critical(f"FATAL: Fallback JSON save also failed: {fallback_e}")
            return False # Return False because the primary save failed

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
        return NetworkxGraphStore()
    else:
        raise ValueError(f"Unknown GRAPH_STORE_TYPE: {store_type}")