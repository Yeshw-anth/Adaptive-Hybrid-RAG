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
    def __init__(self, llm_client: OllamaClient, graph_store: "NetworkxGraphStore"):
        self.llm_client = llm_client
        self.graph_store = graph_store
        self.rule_based_extractor = RuleBasedTripleExtractor()
        self.rebel_extractor = RebelTripleExtractor()
        self.disambiguation_prompt_template = """
You are a knowledge graph expert. Your task is to disambiguate a given entity based on its context.
Provide a concise, canonical name for the entity that includes a parenthetical qualifier to remove ambiguity.

**Instructions:**
1.  Analyze the **Entity** and the **Context** it appeared in.
2.  Create a canonical name. For example, if the entity is "Transformer" and the context is about AI, a good canonical name is "Transformer (AI Architecture)". If the context was about electricity, it might be "Transformer (Electrical Grid)".
3.  Your response MUST be ONLY the canonical name. Do not include any other text, explanations, or summaries.

**Example 1:**
Entity: "Transformer"
Context: "The Transformer architecture, introduced in 'Attention Is All You Need', revolutionized natural language processing by using self-attention mechanisms."
Output:
Transformer (AI Architecture)

**Example 2:**
Entity: "duck"
Context: "The batsman was out for a duck on the first ball."
Output:
Duck (Cricket)

---
Entity: "{entity}"
Context: "{context}"
---
Output:
"""
        self.extraction_prompt_template = """
You are a data extraction expert specializing in building knowledge graphs for software engineering and data science domains.
Your task is to extract knowledge graph triples (subject, predicate, object) from the given text.

**Instructions:**
1.  **Identify Entities:** The subject and object must be specific, named entities (e.g., "RAGOrchestrator", "QueryAnalyzer", "vector database"). Avoid generic terms.
2.  **Use Predefined Predicates:** You MUST use predicates from the following list. Choose the one that best describes the relationship.
    *   `is_a`: For "is a type of" relationships (e.g., ["Vector Database", "is_a", "Database"]).
    *   `has_property`: For describing attributes or properties (e.g., ["FastPipeline", "has_property", "speed-optimized"]).
    *   `uses`: For indicating that one component uses another (e.g., ["RAGOrchestrator", "uses", "QueryAnalyzer"]).
    *   `contributes_to`: For showing a component's role in a larger process (e.g., ["ConfidenceEngine", "contributes_to", "response quality"]).
    *   `is_part_of`: For compositional relationships (e.g., ["QueryAnalyzer", "is_part_of", "Strategy Selection"]).
    *   `enables`: When one component makes another possible (e.g., ["Vector Index", "enables", "semantic search"]).
    *   `manages`: For components that control or orchestrate others (e.g., ["RAGOrchestrator", "manages", "Pipelines"]).
    *   `produces`: For when a component's output is a specific result (e.g., ["QueryAnalyzer", "produces", "QueryMetadata"]).
    *   `related_to`: Use this as a last resort if no other predicate fits.
3.  **Extract Multiple Triples:** Extract as many meaningful and distinct triples as you can from the text.
4.  **Format Output:** Your entire response MUST be a valid JSON list of lists, where each inner list is a triple `[subject, predicate, object]`.
5.  **Empty List for No Triples:** If no triples can be extracted, return an empty list `[]`.

**Example:**
Text: "The RAGOrchestrator uses a QueryAnalyzer to select the best pipeline. This analyzer is part of the strategy selection process and produces detailed QueryMetadata."
Output:
[
  ["RAGOrchestrator", "uses", "QueryAnalyzer"],
  ["QueryAnalyzer", "is_part_of", "Strategy Selection"],
  ["QueryAnalyzer", "produces", "QueryMetadata"]
]

---
Text:
{text}
---
Output:
"""

    async def _disambiguate_entity(self, entity: str, context: str, model: str) -> str:
        """
        Uses an LLM to create a canonical, disambiguated name for an entity.
        """
        if not entity or not context:
            return entity

        prompt = self.disambiguation_prompt_template.format(entity=entity, context=context)
        
        try:
            response_data = await self.llm_client.generate_from_prompt(
                prompt=prompt,
                model=model
            )
            # The response should be just the canonical name, so we strip any extra whitespace
            canonical_name = response_data["content"].strip()
            
            if canonical_name:
                logger.debug(f"Disambiguated '{entity}' to '{canonical_name}'")
                return canonical_name
            else:
                logger.warning(f"LLM returned an empty string for disambiguation of '{entity}'. Using original.")
                return entity
        except Exception as e:
            logger.error(f"An error occurred during entity disambiguation for '{entity}': {e}", exc_info=True)
            # Fallback to the original entity in case of error
            return entity

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
        Generates triples from a batch of texts using a hybrid, three-tiered approach,
        including an entity disambiguation step.
        1. Pass 1: Use the high-precision REBEL model.
        2. Pass 2: For texts with no results, fall back to the rule-based extractor.
        3. Pass 3: For remaining texts, use the general-purpose LLM as a final attempt.
        4. Pass 4: Disambiguate entities in all extracted triples.
        """
        # This list will store tuples of (triple, context_text)
        triples_with_context: List[Tuple[Tuple[str, str, str], str]] = []
        
        # Create a dictionary to track texts that still need processing
        remaining_texts = {i: text for i, text in enumerate(texts)}

        # --- Pass 1: High-Precision REBEL Extraction ---
        logger.info(f"Running Pass 1: REBEL extraction on {len(remaining_texts)} text chunks.")
        texts_for_next_pass = {}
        for index, text in remaining_texts.items():
            rebel_triples = self.rebel_extractor.extract(text)
            if rebel_triples:
                for triple in rebel_triples:
                    triples_with_context.append((triple, text))
                logger.debug(f"REBEL found {len(rebel_triples)} triples from chunk {index}.")
            else:
                # If REBEL finds nothing, mark this text for the next pass
                texts_for_next_pass[index] = text
        remaining_texts = texts_for_next_pass
        
        if not remaining_texts:
            logger.info("REBEL extraction was successful for all chunks. No fallback needed.")
        
        # --- Pass 2: Fast, Rule-Based Fallback ---
        if remaining_texts:
            logger.info(f"Running Pass 2: Rule-based fallback on {len(remaining_texts)} chunks.")
            texts_for_next_pass = {}
            for index, text in remaining_texts.items():
                rule_based_triples = self.rule_based_extractor.extract_triples(text)
                if rule_based_triples:
                    for triple in rule_based_triples:
                        triples_with_context.append((triple, text))
                    logger.debug(f"Rule-based extractor found {len(rule_based_triples)} triples from chunk {index}.")
                else:
                    # If rule-based also finds nothing, mark for the final LLM pass
                    texts_for_next_pass[index] = text
            remaining_texts = texts_for_next_pass

        # --- Pass 3: Slow, General-Purpose LLM Fallback ---
        if remaining_texts:
            logger.info(f"Running Pass 3: LLM fallback on {len(remaining_texts)} chunks.")
            
            # We need to map indices to texts to correctly associate results
            llm_texts_with_indices = list(remaining_texts.items())
            llm_texts = [item[1] for item in llm_texts_with_indices]
            
            # This method now needs to return triples associated with their original text
            llm_triples_with_context = await self._generate_llm_triples(llm_texts, model)
            
            if llm_triples_with_context:
                triples_with_context.extend(llm_triples_with_context)
                logger.info(f"LLM fallback found {len(llm_triples_with_context)} additional triples.")
        
        if not triples_with_context:
            logger.info("No triples were extracted from any source.")
            return []

        # --- Pass 4: Entity Disambiguation ---
        logger.info(f"Running Pass 4: Disambiguating entities for {len(triples_with_context)} triples.")
        disambiguated_triples = await self._disambiguate_triples(triples_with_context, model)

        return list(set(disambiguated_triples))

    async def _disambiguate_triples(
        self,
        triples_with_context: List[Tuple[Tuple[str, str, str], str]],
        model: str
    ) -> List[Tuple[str, str, str]]:
        """
        Disambiguates the subject and object of each triple using the provided context.
        """
        disambiguated_triples = []
        for triple, context in triples_with_context:
            subject, predicate, obj = triple
            
            # Disambiguate subject and object
            disambiguated_subject = await self._disambiguate_entity(subject, context, model)
            disambiguated_object = await self._disambiguate_entity(obj, context, model)
            
            new_triple = (disambiguated_subject, predicate, disambiguated_object)
            disambiguated_triples.append(new_triple)
            logger.debug(f"Original: {(subject, obj)}, Disambiguated: {(disambiguated_subject, disambiguated_object)}")

        return disambiguated_triples

    async def _generate_llm_triples(self, texts: List[str], model: str) -> List[Tuple[Tuple[str, str, str], str]]:
        """
        (Private) Uses an LLM to extract knowledge graph triples from a batch of text chunks.
        This method is designed to be fault-tolerant and returns triples along with their context.
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

        # Step 2: Try to parse the response.
        try:
            json_response = self.llm_client.clean_json_response(response_str)
            list_of_triple_lists = json.loads(json_response)
            
            if not isinstance(list_of_triple_lists, list) or len(list_of_triple_lists) != len(texts):
                logger.warning(f"LLM batch response is not a list of the correct length. Expected {len(texts)}, got {len(list_of_triple_lists)}. Response: {response_str}")
                return []

            results = []
            for i, triples in enumerate(list_of_triple_lists):
                context_text = texts[i]
                if not isinstance(triples, list):
                    logger.warning(f"Item {i} in LLM batch response is not a list: {triples}")
                    continue
                for t in triples:
                    if isinstance(t, list) and len(t) == 3 and all(isinstance(item, str) for item in t):
                        results.append((tuple(t), context_text))
                    else:
                        logger.warning(f"Skipping invalid triple format in batch response: {t}")
            
            logger.info(f"LLM batch extraction successful. Extracted {len(results)} triples. Token usage: {token_usage}")
            return results

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to decode or validate JSON from LLM batch response: {e}\nResponse: '{response_str}'")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM batch triple extraction: {e}", exc_info=True)
            return []