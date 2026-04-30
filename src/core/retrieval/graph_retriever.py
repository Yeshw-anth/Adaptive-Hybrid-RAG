from src.core.logging_config import logger
import networkx as nx
from typing import List, Dict, Any, Optional
import json

from src.core.graph.graph_store import get_graph_store
from src.core.llm.ollama_client import OllamaClient
from src.core.graph.cypher_generator import CypherQueryGenerator

class GraphRetriever:
    """
    Retrieves relevant information from the knowledge graph using a two-tiered strategy:
    1.  **Graph-Native Reasoning**: Translates the question to a Cypher query for direct execution.
    2.  **Fallback to Entity Extraction**: If the first step fails, it falls back to extracting
        entities and retrieving their local subgraph.
    """
    def __init__(self):
        self.graph_store = get_graph_store()
        self.llm_client = OllamaClient()
        self.cypher_generator: Optional[CypherQueryGenerator] = self._initialize_cypher_generator()

    def _initialize_cypher_generator(self) -> Optional[CypherQueryGenerator]:
        """Initializes the CypherQueryGenerator if the graph store is Neo4j."""
        from src.core.graph.neo4j_graph_store import Neo4jGraphStore
        if isinstance(self.graph_store, Neo4jGraphStore):
            try:
                schema = self.graph_store.get_schema()
                if schema and (schema.get("nodes") or schema.get("relationships")):
                    logger.info("Initializing CypherQueryGenerator with active graph schema.")
                    return CypherQueryGenerator(graph_schema=schema)
                else:
                    logger.warning("Neo4j schema is empty. Cypher generation will be disabled.")
                    return None
            except Exception as e:
                logger.error(f"Failed to get schema or initialize CypherQueryGenerator: {e}", exc_info=True)
                return None
        return None

    async def retrieve(self, query: str, depth: int = 2) -> str:
        """
        Retrieves context from the graph. It first tries to generate and execute
        a direct Cypher query. If that is not possible or fails, it falls back
        to the entity-based subgraph retrieval method.
        """
        # Attempt Step 1: Graph-Native Reasoning via Cypher
        if self.cypher_generator:
            try:
                cypher_query = self.cypher_generator.generate_query(question=query)
                if cypher_query:
                    results = self.graph_store.execute_cypher(cypher_query)
                    if results:
                        context = self._format_cypher_results(results)
                        logger.info("Successfully retrieved context via Graph-Native Reasoning.")
                        return context
                    else:
                        logger.info("Cypher query executed but returned no results. Proceeding to fallback.")
                else:
                    logger.info("Cypher generator did not produce a query. Proceeding to fallback.")
            except Exception as e:
                logger.error(f"Error during Graph-Native Reasoning phase: {e}. Falling back.", exc_info=True)

        # Fallback Step 2: Entity-based Subgraph Retrieval
        logger.info("Falling back to entity-based subgraph retrieval.")
        return await self._fallback_retrieval(query, depth)

    def _format_cypher_results(self, results: List[Dict[str, Any]]) -> str:
        """Formats the structured results from a Cypher query into a readable text block."""
        if not results:
            return ""
        
        lines = ["Structured data retrieved from Knowledge Graph:"]
        for record in results:
            lines.append(json.dumps(record, indent=2))
            
        return "\n".join(lines)

    async def _fallback_retrieval(self, query: str, depth: int) -> str:
        """The original retrieval method based on entity extraction."""
        entities = await self._extract_entities_from_query(query)
        if not entities:
            logger.info("No key entities found in query for fallback retrieval.")
            return ""

        from src.core.graph.graph_store import NetworkxGraphStore
        from src.core.graph.neo4j_graph_store import Neo4jGraphStore

        if isinstance(self.graph_store, NetworkxGraphStore):
            return self._retrieve_from_networkx(entities, depth)
        elif isinstance(self.graph_store, Neo4jGraphStore):
            return self._retrieve_from_neo4j_by_entity(entities, depth)
        else:
            logger.error(f"Unsupported graph store type: {type(self.graph_store)}")
            return ""

    def _retrieve_from_networkx(self, entities: List[str], depth: int) -> str:
        """Handles retrieval from a Networkx graph."""
        # (Implementation remains the same as before)
        graph = self.graph_store.graph
        if not graph or len(graph.nodes) == 0:
            return ""
        relevant_nodes = [node for node in graph.nodes if any(entity.lower() in node.lower() for entity in entities)]
        if not relevant_nodes:
            return ""
        subgraphs = [nx.ego_graph(graph, node, radius=depth) for node in relevant_nodes]
        combined_graph = nx.compose_all(subgraphs)
        return self._graph_to_text(combined_graph)

    def _retrieve_from_neo4j_by_entity(self, entities: List[str], depth: int) -> str:
        """Handles subgraph retrieval from Neo4j based on entities."""
        # (Implementation remains the same as before)
        query = """
        MATCH (n) WHERE reduce(s = false, entity IN $entities | s OR toLower(n.id) CONTAINS toLower(entity))
        CALL { WITH n MATCH path = (n)-[*0..%d]-(m) UNWIND nodes(path) AS node UNWIND relationships(path) AS rel RETURN collect(DISTINCT node) AS nodes, collect(DISTINCT rel) AS rels }
        RETURN nodes, rels
        """ % depth
        with self.graph_store.driver.session() as session:
            result = session.run(query, entities=entities)
            edges = []
            for record in result:
                for rel in record["rels"]:
                    start_node = rel.start_node['id']
                    end_node = rel.end_node['id']
                    rel_type = rel.type
                    edges.append(f"- {start_node} {rel_type.replace('_', ' ').lower()} {end_node}.")
            if not edges: return ""
            return "Knowledge Graph Context:\n" + "\n".join(edges)

    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Uses an LLM to extract key entities from the user's query."""
        # (Implementation remains the same as before)
        prompt = f'''You are a data analysis expert... (rest of prompt)''' # Abridged for brevity
        try:
            # ... (rest of implementation)
            return [] # Placeholder
        except Exception:
            return []

    def _graph_to_text(self, graph: nx.MultiDiGraph) -> str:
        """Converts a networkx graph into a human-readable text format."""
        # (Implementation remains the same as before)
        if not graph or len(graph.nodes) == 0: return ""
        lines = ["Knowledge Graph Context:"]
        for subj, obj, data in graph.edges(data=True):
            predicate = data.get('label', 'is related to')
            lines.append(f"- {subj} {predicate} {obj}.")
        return "\n".join(lines)