from src.core.llm.ollama_client import OllamaClient
from src.core.logging_config import logger

class CypherQueryGenerator:
    """
    Uses an LLM to translate a natural language question into a Cypher query
    by providing the graph schema as context.
    """
    def __init__(self, graph_schema: dict):
        self.llm_client = OllamaClient()
        self.graph_schema = self._format_schema_for_prompt(graph_schema)
        self.system_prompt = self._create_system_prompt()

    def _format_schema_for_prompt(self, schema: dict) -> str:
        """
        Formats the graph schema into a string that can be easily understood
        by the LLM in a prompt.
        """
        node_props = []
        for label, props in schema.get("nodes", {}).items():
            props_str = ", ".join([f"{p['propertyName']}: {p['propertyTypes']}" for p in props])
            node_props.append(f"- Node `{label}` has properties: {{{props_str}}}")

        rel_props = []
        for rel_type, props in schema.get("relationships", {}).items():
            props_str = ", ".join([f"{p['propertyName']}: {p['propertyTypes']}" for p in props])
            rel_props.append(f"- Relationship `{rel_type}` has properties: {{{props_str}}}")

        return "\n".join(node_props) + "\n\n" + "\n".join(rel_props)

    def _create_system_prompt(self) -> str:
        """
        Creates the system prompt for the LLM, instructing it on how to
        behave as a Cypher query generator.
        """
        return f"""
You are an expert Neo4j Cypher query translator. Your role is to convert natural language questions into precise and executable Cypher queries.

You will be given a question and the schema of the graph database. You must use the provided schema to construct the query.

**Graph Schema:**
{self.graph_schema}

**Instructions:**
1.  **Analyze the Question:** Understand the user's intent and the entities and relationships involved.
2.  **Use the Schema:** Only use the node labels, relationship types, and properties defined in the schema. Do not invent new ones.
3.  **Return ONLY the Query:** Your response must be a single, clean Cypher query. Do not include any explanations, comments, or markdown formatting like ```cypher ... ```.
4.  **Handle Ambiguity:** If a question is ambiguous, make a reasonable assumption based on the schema.
5.  **Be Precise:** Pay close attention to property names and relationship directions.

Example:
- Question: "What are the latest documents about 'Project Phoenix'?"
- Your Response:
MATCH (d:Document)-[:MENTIONS]->(e:Entity {{name: 'Project Phoenix'}})
RETURN d.title, d.source
ORDER BY d.last_modified DESC
LIMIT 5
"""

    def generate_query(self, question: str) -> str:
        """
        Generates a Cypher query from a natural language question.
        """
        logger.info(f"Generating Cypher query for question: '{question}'")
        
        response = self.llm_client.generate(
            prompt=question,
            system_prompt=self.system_prompt
        )
        
        # Clean up the response to ensure it's just the query
        cypher_query = response.strip()
        if cypher_query.startswith("```cypher"):
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
        logger.info(f"Generated Cypher query: {cypher_query}")
        return cypher_query