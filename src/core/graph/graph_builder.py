import json
from src.core.logging_config import logger
from typing import List, Tuple
from src.core.llm.ollama_client import OllamaClient
from pydantic import ValidationError
from src.core.graph.validation import ValidTriple
from src.core.graph.rule_based_extractor import RuleBasedTripleExtractor
from src.core.graph.rebel_extractor import RebelTripleExtractor

class KnowledgeGraphBuilder:
    """
    Extracts entities and relationships from text to build a knowledge graph.
    """
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        self.rule_based_extractor = RuleBasedTripleExtractor()
        self.rebel_extractor = RebelTripleExtractor()
        self.extraction_prompt_template = """
You are a data extraction expert. Your task is to extract knowledge graph triples (subject, predicate, object) from the given text.
- The subject and object should be specific entities.
- The predicate should describe the relationship between them.
- Extract as many meaningful triples as you can.
- Output the triples in a valid JSON list format, where each item is a list of three strings: [subject, predicate, object].
- If no triples can be extracted, return an empty list.

Example:
Text: "The RAGOrchestrator uses a QueryAnalyzer to select the best pipeline."
Output:
[
  ["RAGOrchestrator", "uses", "QueryAnalyzer"],
  ["QueryAnalyzer", "is used by", "RAGOrchestrator"],
  ["RAGOrchestrator", "selects", "pipeline"]
]

Text:
{text}

Output:
"""

    async def generate_triples(self, text: str, model: str) -> List[Tuple[str, str, str]]:
        """
        Uses an LLM to extract knowledge graph triples from a text chunk.
        """
        if not text or not text.strip():
            return []

        prompt = self.extraction_prompt_template.format(text=text)
        
        response_str = ""  # Initialize in case of early exit
        try:
            # The client now returns a dictionary with 'content' and 'token_usage'
            response_data = await self.llm_client.generate_from_prompt(
                prompt=prompt,
                model=model
            )
            response_str = response_data["content"]
            token_usage = response_data["token_usage"]

            json_response = self.llm_client.clean_json_response(response_str)
            
            triples = json.loads(json_response)
            
            validated_triples: List[Tuple[str, str, str]] = []
            if not isinstance(triples, list):
                logger.warning(f"LLM response is not a list: {triples}")
                return []

            for t in triples:
                if isinstance(t, list) and len(t) == 3 and all(isinstance(i, str) for i in t):
                    validated_triples.append((t[0], t[1], t[2]))
                else:
                    logger.warning(f"Skipping invalid triple format: {t}")

            logger.info(
                f"Extracted {len(validated_triples)} triples. "
                f"Token usage: {token_usage['input_tokens']} (in), {token_usage['output_tokens']} (out)."
            )
            return validated_triples

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to decode or validate JSON from LLM response: {e}\nResponse: '{response_str}'")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during triple extraction: {e}", exc_info=True)
            return []

    async def generate_triples_batch(self, texts: List[str], model: str) -> List[Tuple[str, str, str]]:
        """
        Generates triples from a batch of texts using a hybrid, three-tiered approach.
        1. Pass 1: Use the high-precision REBEL model.
        2. Pass 2: For texts with no results, fall back to the rule-based extractor.
        3. Pass 3: For remaining texts, use the general-purpose LLM as a final attempt.
        """
        all_triples: List[Tuple[str, str, str]] = []
        
        # Create a dictionary to track texts that still need processing
        remaining_texts = {i: text for i, text in enumerate(texts)}

        # --- Pass 1: High-Precision REBEL Extraction ---
        logger.info(f"Running Pass 1: REBEL extraction on {len(remaining_texts)} text chunks.")
        texts_for_next_pass = {}
        for index, text in remaining_texts.items():
            rebel_triples = self.rebel_extractor.extract(text)
            if rebel_triples:
                all_triples.extend(rebel_triples)
                logger.debug(f"REBEL found {len(rebel_triples)} triples from chunk {index}.")
            else:
                # If REBEL finds nothing, mark this text for the next pass
                texts_for_next_pass[index] = text
        remaining_texts = texts_for_next_pass
        
        if not remaining_texts:
            logger.info("REBEL extraction was successful for all chunks. No fallback needed.")
            return list(set(all_triples))

        # --- Pass 2: Fast, Rule-Based Fallback ---
        logger.info(f"Running Pass 2: Rule-based fallback on {len(remaining_texts)} chunks.")
        texts_for_next_pass = {}
        for index, text in remaining_texts.items():
            rule_based_triples = self.rule_based_extractor.extract_triples(text)
            if rule_based_triples:
                all_triples.extend(rule_based_triples)
                logger.debug(f"Rule-based extractor found {len(rule_based_triples)} triples from chunk {index}.")
            else:
                # If rule-based also finds nothing, mark for the final LLM pass
                texts_for_next_pass[index] = text
        remaining_texts = texts_for_next_pass

        # --- Pass 3: Slow, General-Purpose LLM Fallback ---
        if not remaining_texts:
            logger.info("Rule-based fallback was successful for all remaining chunks. No LLM needed.")
            return list(set(all_triples))

        logger.info(f"Running Pass 3: LLM fallback on {len(remaining_texts)} chunks.")
        
        # We pass the original indices to map the results back correctly
        indices_for_llm = list(remaining_texts.keys())
        texts_for_llm = list(remaining_texts.values())
        
        llm_triples = await self._generate_llm_triples(texts_for_llm, model)
        all_triples.extend(llm_triples)
        
        return list(set(all_triples))

    async def _generate_llm_triples(self, texts: List[str], model: str) -> List[Tuple[str, str, str]]:
        """
        (Private) Uses an LLM to extract knowledge graph triples from a batch of text chunks.
        This method is designed to be fault-tolerant. If the LLM fails to return valid
        JSON for a batch, it logs a warning and returns an empty list for that batch,
        preventing one failure from stopping the entire ingestion process.
        """
        if not texts:
            return []

        formatted_texts = "\n\n".join([f"Text {i+1}: {text}" for i, text in enumerate(texts)])
        
        batch_prompt_template = f"""
You are a data extraction expert. Your ONLY job is to extract knowledge graph triples (subject, predicate, object) from the multiple text chunks provided below.

- You will be given N text chunks.
- Your response MUST be a JSON list containing exactly N elements.
- Each element in the list MUST be a list of the triples you extracted from the corresponding text chunk.
- Each triple MUST be a list of three strings: [subject, predicate, object].
- If no triples are found in a chunk, the element for that chunk MUST be an empty list `[]`.
- CRITICAL: Your entire response MUST be ONLY the top-level JSON list. Do not include any other text, explanations, or summaries.

Example:
Input Texts:
Text 1: "The RAGOrchestrator uses a QueryAnalyzer."
Text 2: "No relevant information here."
Text 3: "The IngestionPipeline processes documents."

Output:
[
  [["RAGOrchestrator", "uses", "QueryAnalyzer"]],
  [],
  [["IngestionPipeline", "processes", "documents"]]
]
---
Texts:
{formatted_texts}
---
Output:
"""
        
        # Step 1: Get response from LLM
        try:
            response_data = await self.llm_client.generate_from_prompt(
                prompt=batch_prompt_template,
                model=model
            )
            response_str = response_data["content"]
            token_usage = response_data.get("token_usage", {})
        except Exception as e:
            logger.error(f"An unexpected error occurred during the initial LLM call for batch: {e}", exc_info=True)
            return []

        # Step 2: Try to parse the response. If it fails, attempt self-correction.
        try:
            json_response = self.llm_client.clean_json_response(response_str)
            list_of_triple_lists = json.loads(json_response)
            
            if not isinstance(list_of_triple_lists, list) or len(list_of_triple_lists) != len(texts):
                logger.warning(f"LLM batch response is not a list of the correct length. Expected {len(texts)}, got {len(list_of_triple_lists)}. Response: {response_str}")
                return []

            validated_triples = []
            for i, triples in enumerate(list_of_triple_lists):
                if not isinstance(triples, list):
                    logger.warning(f"Item {i} in LLM batch response is not a list: {triples}")
                    continue
                for t in triples:
                    if isinstance(t, list) and len(t) == 3 and all(isinstance(item, str) for item in t):
                        validated_triples.append(tuple(t))
                    else:
                        logger.warning(f"Skipping invalid triple format in batch response: {t}")
            
            logger.info(f"LLM batch extraction successful. Extracted {len(validated_triples)} triples. Token usage: {token_usage}")
            return validated_triples

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to decode or validate JSON from LLM batch response: {e}\nResponse: '{response_str}'")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM batch triple extraction: {e}", exc_info=True)
            return []