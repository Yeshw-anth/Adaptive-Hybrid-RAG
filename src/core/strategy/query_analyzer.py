import logging
import json
import re
from typing import Literal, List, Dict
from pydantic import BaseModel, Field, ValidationError
from enum import Enum

from src.core.llm.ollama_client import OllamaClient
from src.data.schemas import QueryMetadata
from src.config import settings

logger = logging.getLogger(__name__)

# --- Pydantic Models for Strict Validation ---

class Intent(str, Enum):
    FACT_SEEKING = "fact-seeking"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    CAUSAL_ANALYSIS = "causal-analysis"

class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ExpectedAnswerFormat(str, Enum):
    LIST = "list"
    SINGLE_VALUE = "single_value"
    EXPLANATION = "explanation"
    CODE_SNIPPET = "code_snippet"
    TABLE = "table"

class QueryType(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    KEYWORD = "keyword"

class LLMAnalysisResponse(BaseModel):
    intent: Intent = Field(..., description="The user's primary goal.")
    complexity: Complexity = Field(..., description="The query's analytical complexity.")
    keywords: List[str] = Field(..., description="A list of 3-5 essential keywords.")
    expected_answer_format: ExpectedAnswerFormat = Field(..., description="The likely format for the answer.")
    query_type: QueryType = Field(..., description="The type of the query.")

class QueryAnalyzer:
    """
    Analyzes a query to extract metadata. It uses a heuristic to quickly identify
    keyword queries and an LLM for more complex, natural language queries.
    """

    def __init__(self, llm_wrapper: OllamaClient, max_retries: int = 2):
        self.llm_wrapper = llm_wrapper
        self.max_retries = max_retries
        self.question_words = {"who", "what", "when", "where", "why", "how", "which", "whom", "whose"}
        self.prompt_template = """Analyze the user query below. Your task is to extract key attributes and respond ONLY with a valid JSON object. Do not include any other text, explanations, or markdown code fences.

**JSON Schema:**
{{
  "intent": "{intents}",
  "complexity": "{complexities}",
  "keywords": ["string", "string", ...],
  "expected_answer_format": "{formats}",
  "query_type": "{query_types}"
}}

**Instructions:**
1.  **intent**: Choose the single best fit from the allowed values.
2.  **complexity**: Choose the single best fit from the allowed values.
3.  **keywords**: Extract 3-5 essential nouns, verbs, or named entities.
4.  **expected_answer_format**: Choose the most likely format the user wants.
5.  **query_type**: Choose the single best fit from the allowed values.

**User Query:** "{query}"

**Your JSON Response:**
"""
        self.retry_prompt_template = """Your previous response was not valid JSON. Please correct it.
Original Query: "{query}"
Error: {error}
Respond with ONLY the corrected JSON object.
"""

    def _is_keyword_query(self, query: str) -> bool:
        """
        A heuristic to quickly determine if a query is likely a keyword search.
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # Rule 1: Short query length
        if len(words) <= 5:
            # Rule 2: Does not start with a common question word
            if not any(query_lower.startswith(word) for word in self.question_words):
                logger.info(f"Query '{query}' classified as KEYWORD search by heuristic.")
                return True
        return False

    async def analyze(self, query: str, model_override: str = None) -> QueryMetadata:
        """
        Analyzes the query, first using a heuristic for keyword searches,
        then falling back to an LLM for more complex queries.
        """
        if self._is_keyword_query(query):
            return QueryMetadata(
                query=query,
                intent="fact-seeking",  # Default for keyword
                complexity="low",       # Default for keyword
                keywords=query.split(),
                expected_answer_format="single_value", # Default for keyword
                query_type="keyword", # Explicitly set type
                suggested_depth=3,
                content_hints=[]
            )

        model_to_use = model_override or settings.DEFAULT_LLM_MODEL
        
        # Format the prompt in a single step to handle all placeholders correctly
        base_prompt = self.prompt_template.format(
            intents=", ".join(f'"{i.value}"' for i in Intent),
            complexities=", ".join(f'"{c.value}"' for c in Complexity),
            formats=", ".join(f'"{f.value}"' for f in ExpectedAnswerFormat),
            query_types=", ".join(f'"{qt.value}"' for qt in QueryType),
            query=query
        )
        
        for attempt in range(self.max_retries + 1):
            prompt = base_prompt
            if attempt > 0:
                logger.warning(f"Query analysis failed on attempt {attempt}. Retrying...")
                prompt = self.retry_prompt_template.format(query=query, error=self.last_error)
            
            logger.debug(f"Formatted prompt for LLM analysis (Attempt {attempt + 1}):\n{prompt}")

            try:
                response_text = await self.llm_wrapper.generate_from_prompt(prompt, model=model_to_use)
                
                # Use the new robust parsing function
                json_content = self._extract_and_parse_json(response_text)
                llm_response = LLMAnalysisResponse.parse_obj(json_content)
                
                metadata = QueryMetadata(
                    query=query,
                    intent=llm_response.intent.value,
                    complexity=llm_response.complexity.value,
                    keywords=llm_response.keywords,
                    expected_answer_format=llm_response.expected_answer_format.value,
                    query_type=llm_response.query_type.value,
                    suggested_depth=4,
                    content_hints=[]
                )
                
                logger.info(f"Query analysis successful: {metadata.model_dump_json(indent=2)}")
                return metadata

            except (ValidationError, json.JSONDecodeError) as e:
                self.last_error = str(e)
                logger.error(f"Attempt {attempt + 1}: Failed to parse or validate LLM response. Error: {self.last_error}")
                if attempt == self.max_retries:
                    logger.critical("Query analysis failed after multiple retries. Falling back to keyword search.")
                    return QueryMetadata(
                        query=query,
                        intent="fact-seeking",
                        complexity="low",
                        keywords=query.split(),
                        expected_answer_format="single_value",
                        query_type="keyword",
                        suggested_depth=3,
                        content_hints=[]
                    )
            except Exception as e:
                logger.error(f"An unexpected error occurred during query analysis: {e}", exc_info=True)
                if attempt == self.max_retries:
                    logger.critical("Query analysis failed due to an unexpected error. Falling back to keyword search.")
                    return QueryMetadata(
                        query=query,
                        intent="fact-seeking",
                        complexity="low",
                        keywords=query.split(),
                        expected_answer_format="single_value",
                        query_type="keyword",
                        suggested_depth=3,
                        content_hints=[]
                    )

        # This part should ideally not be reached, but as a final safeguard:
        logger.error("Fell through query analysis loop. This should not happen.")
        return QueryMetadata(
            query=query,
            intent="fact-seeking",
            complexity="low",
            keywords=query.split(),
            expected_answer_format="single_value",
            query_type="keyword",
            suggested_depth=3,
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