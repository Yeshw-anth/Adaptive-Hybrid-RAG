import logging
import json
from typing import List
from llm.llm_wrapper import LLMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Expands a user query into multiple variations to improve retrieval recall.
    This is triggered when the initial retrieval results have medium confidence.
    """

    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm_wrapper = llm_wrapper
        self.prompt_template = """
        You are an expert at query expansion. Your task is to generate 3 diverse, alternative phrasings for the given user query.
        The phrasings should explore different facets, synonyms, and user intents related to the original query.
        Return the results as a JSON list of strings.

        Original Query: "{query}"

        JSON Output:
        """

    def expand(self, query: str) -> List[str]:
        """
        Generates alternative phrasings for a given query using an LLM.

        Args:
            query: The original user query.

        Returns:
            A list of unique expanded queries, including the original.
        """
        prompt = self.prompt_template.format(query=query)
        
        try:
            response_text = self.llm_wrapper.generate(prompt)
            
            # Clean the response to ensure it is valid JSON
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            
            expanded_queries = json.loads(response_text)

            if not isinstance(expanded_queries, list):
                raise ValueError("LLM response is not a list.")

            # Combine and deduplicate
            all_queries = [query] + expanded_queries
            unique_queries = list(dict.fromkeys(all_queries)) # Preserve order and deduplicate
            
            logger.info(f"Expanded query '{query}' into: {unique_queries}")
            return unique_queries

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response for query expansion '{query}': {e}")
            return [query] # Fallback to the original query
        except Exception as e:
            logger.error(f"An unexpected error occurred during query expansion: {e}")
            return [query]