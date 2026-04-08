import logging
from typing import List

from src.core.llm.ollama_client import OllamaClient
from src.data.schemas import QueryMetadata

logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Intelligently expands a query using an LLM, leveraging query analysis metadata
    to generate more targeted and diverse alternative formulations.
    """

    def __init__(self, llm_wrapper: OllamaClient):
        self.llm_wrapper = llm_wrapper
        self.prompt_template = """You are a search expert. Your task is to reformulate a user's query to improve search results.
Analyze the original query and its associated metadata to generate 2 diverse and effective alternative queries.

**Original Query:** "{query}"

**Query Analysis:**
- **Intent:** {intent}
- **Complexity:** {complexity}
- **Keywords:** {keywords}
- **Expected Answer Format:** {expected_answer_format}

Based on this analysis, generate two alternative queries.
- One query should be a more specific version of the original, using the keywords.
- The other should be a broader, more conceptual query related to the user's intent.
- Return ONLY the queries, one per line.

Example:
Original Query: "what is the performance of the new model"
Analysis: {{'intent': 'fact-seeking', 'keywords': ['performance', 'new model']}}
Alternative Queries:
How is the new model's performance measured?
Broader impact of the new model's capabilities.

**Your Turn:**

Original Query: "{query}"
Query Analysis:
- **Intent:** {intent}
- **Complexity:** {complexity}
- **Keywords:** {keywords}
- **Expected Answer Format:** {expected_answer_format}

Alternative Queries:
"""

    async def expand(self, query: str, query_metadata: QueryMetadata) -> list[str]:
        prompt = self.prompt_template.format(
            query=query,
            intent=query_metadata.intent,
            complexity=query_metadata.complexity,
            keywords=", ".join(query_metadata.keywords),
            expected_answer_format=query_metadata.expected_answer_format
        )
        response = await self.llm_wrapper.generate_from_prompt(prompt)
        
        # Strict parsing to remove junk lines and numbering
        queries = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove potential numbering like "1. " or "- "
            if line.startswith(tuple(f"{i}." for i in range(10))) or line.startswith("-"):
                line = line.split(" ", 1)[-1]
            queries.append(line.strip('"'))

        logger.info(f"Intelligently expanded query '{query}' into: {queries}")
        return queries[:2]