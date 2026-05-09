from src.core.logging_config import logger
import networkx as nx
from typing import List, Dict, Any, Optional
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.core.graph.graph_store import get_graph_store
from src.core.llm.ollama_client import OllamaClient
from src.core.graph.cypher_generator import CypherQueryGenerator

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except Exception as e:
    logger.error(f"An error occurred during NLTK data check/download: {e}")
    # Attempt to download all essential packages, as we can't be sure which one failed.
    logger.info("Attempting to download all required NLTK packages...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK packages downloaded successfully.")
    except Exception as download_e:
        logger.critical(f"Failed to download essential NLTK packages: {download_e}. The application may not function correctly.")
        # Depending on the application's requirements, you might want to raise the exception here to halt execution.
        # raise download_e

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

    def _normalize_entity(self, entity: str) -> str:
        """
        Normalizes entity names for consistency in the graph using lemmatization.
        - Converts to lowercase
        - Tokenizes
        - Lemmatizes each token
        - Removes stop words (except for very short entities)
        - Strips leading/trailing whitespace
        """
        entity = entity.lower().strip()
        tokens = word_tokenize(entity)
        
        # Lemmatize tokens
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stop words, but keep them if the entity is just a stop word (e.g., "IT")
        if len(lemmatized_tokens) > 1:
            lemmatized_tokens = [token for token in lemmatized_tokens if token not in stop_words]
        
        # Re-join and strip
        normalized_entity = ' '.join(lemmatized_tokens).strip()
        
        # If normalization results in an empty string (e.g., entity was only stop words),
        # return the original cleaned entity.
        return normalized_entity if normalized_entity else entity

    async def _fallback_retrieval(self, query: str, depth: int) -> str:
        """The original retrieval method based on entity extraction."""
        entities = await self._extract_entities_from_query(query)
        if not entities:
            logger.info("No key entities found in query for fallback retrieval.")
            return ""

        normalized_entities = [self._normalize_entity(e) for e in entities]
        logger.info(f"Normalized entities for graph lookup: {normalized_entities}")

        from src.core.graph.graph_store import NetworkxGraphStore
        from src.core.graph.neo4j_graph_store import Neo4jGraphStore

        if isinstance(self.graph_store, NetworkxGraphStore):
            return self._retrieve_from_networkx(normalized_entities, depth)
        elif isinstance(self.graph_store, Neo4jGraphStore):
            return self._retrieve_from_neo4j_by_entity(normalized_entities, depth)
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
        prompt = f'''
        You are an expert in knowledge graph construction and querying. Your task is to extract the key entities from the following user query. 
        Focus on identifying specific nouns, concepts, or proper names that are likely to be nodes in a knowledge graph.
        Return the entities as a JSON list of strings. If no entities are found, return an empty list.

        Example:
        Query: "What is the relationship between graph databases and vector databases?"
        Output: ["graph databases", "vector databases"]

        Query: "{query}"
        Output:
        '''
        try:
            # 1. Generate raw text response
            response_dict = await self.llm_client.generate_from_prompt(prompt)
            raw_text = response_dict.get("content", "")

            if not raw_text:
                logger.warning("LLM returned an empty response for entity extraction.")
                return []

            # 2. Clean the text to get a valid JSON string
            cleaned_json_str = self.llm_client.clean_json_response(raw_text)
            
            # 3. Parse the JSON string
            entities = json.loads(cleaned_json_str)

            if isinstance(entities, list):
                logger.info(f"Extracted entities: {entities}")
                return entities
            else:
                logger.warning(f"LLM returned a non-list for entity extraction: {entities}")
                return []
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing JSON for entity extraction: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during entity extraction: {e}", exc_info=True)
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