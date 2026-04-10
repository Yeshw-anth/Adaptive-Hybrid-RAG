import logging
import json
import re
from typing import Literal, List, Dict, get_args
from pydantic import BaseModel, Field, ValidationError

from src.core.llm.ollama_client import OllamaClient
from src.data.schemas import QueryMetadata
from src.config import settings

logger = logging.getLogger(__name__)

# --- Pydantic Models for Strict Validation ---

Intent = Literal["fact-seeking", "summary", "comparison", "causal-analysis"]
Complexity = Literal["low", "medium", "high"]
ExpectedAnswerFormat = Literal["list", "single_value", "explanation", "code_snippet", "table"]
QueryType = Literal["simple", "complex", "analytical", "comparative", "keyword"]


class LLMAnalysisResponse(BaseModel):
    intent: Intent = Field(..., description="The user's primary goal.")
    complexity: Complexity = Field(..., description="The query's analytical complexity.")
    keywords: List[str] = Field(..., description="A list of 3-5 essential keywords.")
    expected_answer_format: ExpectedAnswerFormat = Field(..., description="The likely format for the answer.")
    query_type: QueryType = Field(..., description="The type of the query.")

class QueryAnalyzer:
    """
    Analyzes a query to extract metadata using a sophisticated LLM prompt
    that encourages chain-of-thought reasoning to select the best strategy.
    """

    def __init__(self, llm_wrapper: OllamaClient, max_retries: int = 2):
        self.llm_wrapper = llm_wrapper
        self.max_retries = max_retries
        self.last_error = ""
        self.system_prompt = """You are an expert query analyzer for an advanced RAG system. Your task is to analyze the user's query, reason about the best strategy, and then output a single, valid JSON object.

Here are the available RAG strategies and their use cases:

1.  **`keyword`**:
    *   **Use Case**: For simple, specific, fact-based lookups. Ideal for queries that look like search engine terms.
    *   **Characteristics**: Short, few words, often contains proper nouns, model numbers, or specific terms. Lacks conversational language.
    *   **Examples**: "Q3 2023 financial report", "llama2-7b context window", "system design for microservices"

2.  **`fast`**:
    *   **Use Case**: For simple questions that can likely be answered from a single piece of context. Speed is a priority.
    *   **Characteristics**: Conversational but straightforward. Asks "what is" or "who is".
    *   **Examples**: "What is the capital of France?", "Who wrote 'The Great Gatsby'?"

3.  **`accurate`**:
    *   **Use Case**: For complex, nuanced, or comparative questions that require synthesizing information from multiple sources. Accuracy is the top priority.
    *   **Characteristics**: Asks for comparisons ("compare", "vs"), analysis ("why", "how"), or summaries of broad topics.
    *   **Examples**: "Compare the performance of GPT-4 and Claude 3 Opus.", "What are the main arguments for and against universal basic income?", "Summarize the plot of 'Dune'."

4.  **`code`**:
    *   **Use Case**: For queries that explicitly ask for code, programming concepts, or software development help.
    *   **Characteristics**: Mentions programming languages, libraries, algorithms, or development concepts.
    *   **Examples**: "Show me a Python example of a class", "How to use the requests library in Go?", "What is the time complexity of quicksort?"

Your reasoning should be based on these definitions. Your final output MUST be a single JSON object, with no other text.
"""
        self.user_prompt_template = """Analyze the following query and provide a single, valid JSON response.

**Query:** "{query}"

**JSON Response Format:**
```json
{{
    "intent": "MUST be one of: {intents}",
    "complexity": "MUST be one of: {complexities}",
    "keywords": ["list", "of", "keywords"],
    "expected_answer_format": "MUST be one of: {formats}",
    "query_type": "MUST be one of: {query_types}"
}}
```"""
        self.retry_prompt_template = """Your previous response was not valid JSON. Please correct it.
Original Query: "{query}"
Error: {error}
Respond with ONLY the corrected JSON object inside a `json` block.
"""

    async def analyze(self, query: str, model_override: str = None) -> QueryMetadata:
        """
        Analyzes the query using a sophisticated LLM chain-of-thought prompt.
        """
        model_to_use = model_override or settings.DEFAULT_LLM_MODEL
        
        # This prompt is now much more detailed and guides the LLM better.
        user_prompt = self.user_prompt_template.format(
            query=query,
            intents=", ".join(f'"{i}"' for i in get_args(Intent)),
            complexities=", ".join(f'"{c}"' for c in get_args(Complexity)),
            formats=", ".join(f'"{f}"' for f in get_args(ExpectedAnswerFormat)),
            query_types=", ".join(f'"{qt}"' for qt in get_args(QueryType))
        )
        
        for attempt in range(self.max_retries + 1):
            prompt_for_llm = user_prompt
            if attempt > 0:
                logger.warning(f"Query analysis failed on attempt {attempt}. Retrying...")
                prompt_for_llm = self.retry_prompt_template.format(query=query, error=self.last_error)
            
            logger.debug(f"Formatted prompt for LLM analysis (Attempt {attempt + 1}):\n{prompt_for_llm}")

            try:
                # Use the new method that accepts a system prompt
                response_text = await self.llm_wrapper.generate_with_system_prompt(
                    system_prompt=self.system_prompt,
                    user_prompt=prompt_for_llm,
                    model=model_to_use
                )
                
                json_content = self._extract_and_parse_json(response_text)
                llm_response = LLMAnalysisResponse.parse_obj(json_content)
                
                metadata = QueryMetadata(
                    query=query,
                    intent=llm_response.intent,
                    complexity=llm_response.complexity,
                    keywords=llm_response.keywords,
                    expected_answer_format=llm_response.expected_answer_format,
                    query_type=llm_response.query_type
                )
                
                logger.info(f"Query analysis successful: {metadata.model_dump_json(indent=2)}")
                return metadata

            except (ValidationError, json.JSONDecodeError) as e:
                self.last_error = str(e)
                logger.error(f"Attempt {attempt + 1}: Failed to parse or validate LLM response. Error: {self.last_error}")
                if attempt == self.max_retries:
                    logger.critical("Query analysis failed after multiple retries. Falling back to a default 'accurate' strategy.")
                    return self._fallback_metadata(query)
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"An unexpected error occurred during query analysis: {e}", exc_info=True)
                if attempt == self.max_retries:
                    logger.critical("Query analysis failed due to an unexpected error. Falling back to a default 'accurate' strategy.")
                    return self._fallback_metadata(query)

        # This part should ideally not be reached, but as a final safeguard:
        logger.error("Fell through query analysis loop. This should not happen.")
        return self._fallback_metadata(query)

    def _fallback_metadata(self, query: str) -> QueryMetadata:
        """Provides a safe, default metadata object when analysis fails."""
        return QueryMetadata(
            query=query,
            intent="fact-seeking",
            complexity="high", # Assume high complexity on failure
            keywords=query.split(),
            expected_answer_format="explanation",
            query_type="complex", # Default to complex to trigger 'accurate' pipeline
            suggested_depth=4,
            content_hints=[]
        )

    def _extract_and_parse_json(self, text: str) -> Dict:
        """
        Extracts a JSON object from a string, cleans it, and parses it.
        Handles markdown fences and attempts to fix common JSON errors.
        """
        # Regex to find JSON content, including within markdown fences
        match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', text, re.DOTALL)
        if not match:
            logger.warning("No JSON object found in the LLM response.")
            raise json.JSONDecodeError("No JSON found", text, 0)

        json_str = match.group(1) or match.group(2)
        
        try:
            # First attempt to parse directly
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}. Attempting to fix...")
            # Attempt to fix common errors (e.g., trailing commas, single quotes)
            # This is a simple fix; more complex ones could be added.
            fixed_json_str = json_str.replace("'", '"').rstrip().rstrip(',')
            # Try to re-parse after fixing
            try:
                return json.loads(fixed_json_str)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON even after attempting to fix it.")
                raise # Re-raise the original error to be caught by the main loop