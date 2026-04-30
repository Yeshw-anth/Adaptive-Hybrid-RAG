import logging
from neo4j import GraphDatabase, Driver
from src.config.settings import settings
from src.core.logging_config import logger

class Neo4jGraphStore:
    """
    Manages the connection and operations with a Neo4j database.
    """
    def __init__(self, uri: str = settings.NEO4J_URI, user: str = settings.NEO4J_USER, password: str = settings.NEO4J_PASSWORD):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Driver = None
        self._connect()
        self._create_constraints()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            raise

    def _create_constraints(self):
        """
        Ensures that node IDs are unique to prevent duplicate nodes.
        This is a critical best practice for graph data modeling.
        """
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT unique_node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
                logger.info("Unique node ID constraint ensured in Neo4j.")
            except Exception as e:
                logger.error(f"Failed to create unique node ID constraint: {e}", exc_info=True)

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def add_node(self, node_id: str, **kwargs):
        """
        Adds or updates a node in the graph.
        Uses MERGE to prevent creating duplicate nodes.
        """
        with self.driver.session() as session:
            # The 'id' property is set from the node_id argument
            properties = {"id": node_id, **kwargs}
            
            # Use MERGE on the 'id' property to find or create the node
            # Use SET to add/update all other properties
            session.run(
                "MERGE (n:Node {id: $id}) SET n += $props",
                id=node_id,
                props=properties
            )
            logger.debug(f"Added/updated node {node_id} in Neo4j.")

    def add_edge(self, source_id: str, target_id: str, **kwargs):
        """
        Adds an edge between two nodes.
        The 'label' for the relationship is taken from the kwargs.
        """
        with self.driver.session() as session:
            relationship_type = kwargs.get("label", "RELATED_TO").replace(" ", "_").upper()
            properties = {key: value for key, value in kwargs.items() if key != "label"}

            session.run(
                """
                MATCH (a:Node {id: $source_id})
                MATCH (b:Node {id: $target_id})
                MERGE (a)-[r:%s]->(b)
                SET r += $props
                """ % relationship_type,  # Use %-formatting for the relationship type
                source_id=source_id,
                target_id=target_id,
                props=properties
            )
            logger.debug(f"Added edge from {source_id} to {target_id} in Neo4j.")

    def query_graph(self, query: str):
        """
        Performs a simple text search across node properties.
        This is a basic search and can be expanded with full-text indexing.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:Node)
                WHERE any(key IN keys(n) WHERE toString(n[key]) CONTAINS $query)
                RETURN n
                """,
                query=query
            )
            return [record["n"] for record in result]

    def get_all_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n:Node) RETURN n")
            return [record["n"] for record in result]

    def get_all_edges(self):
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN r")
            return [record["r"] for record in result]

    def get_schema(self) -> dict:
        """
        Retrieves the graph schema, including node labels, relationship types,
        and properties. This is crucial context for the LLM to generate
        accurate Cypher queries.
        """
        with self.driver.session() as session:
            # This query gets all node labels and their properties
            nodes_schema = session.run("""
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes
                RETURN nodeType, collect({propertyName: propertyTypes}) AS properties
            """).data()

            # This query gets all relationship types and their properties
            rels_schema = session.run("""
                CALL db.schema.relTypeProperties()
                YIELD relType, propertyName, propertyTypes
                // Format the relationship type for better readability
                WITH replace(relType, '`', '') AS relType, propertyName, propertyTypes
                RETURN relType, collect({propertyName: propertyTypes}) AS properties
            """).data()

            schema = {
                "nodes": {item['nodeType']: item['properties'] for item in nodes_schema},
                "relationships": {item['relType']: item['properties'] for item in rels_schema}
            }
            logger.info(f"Retrieved graph schema: {schema}")
            return schema

    def execute_cypher(self, query: str) -> list:
        """
        Executes a given Cypher query directly against the database.
        """
        with self.driver.session() as session:
            try:
                result = session.run(query)
                data = result.data()
                logger.info(f"Successfully executed Cypher query. Found {len(data)} results.")
                return data
            except Exception as e:
                logger.error(f"Failed to execute Cypher query: {query}\nError: {e}", exc_info=True)
                return []

    def save_graph(self):
        """
        In Neo4j, data is saved transactionally, so this method is not needed
        in the same way as the file-based networkx store.
        """
        logger.info("Neo4j is a transactional database; data is saved automatically.")
        pass