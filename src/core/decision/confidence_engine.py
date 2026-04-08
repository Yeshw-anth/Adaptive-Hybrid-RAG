from typing import List, Dict, Any, Optional
import numpy as np
import logging
import json
from pydantic import BaseModel, Field
from src.core.llm.ollama_client import OllamaClient
from src.config import settings

logger = logging.getLogger(__name__)

# --- Pydantic Models for Groundedness Validation ---
class GroundednessCheck(BaseModel):
    is_grounded: bool = Field(..., description="Is the entire answer fully supported by the context?")
    unsupported_claims: List[str] = Field(..., description="A list of specific claims in the answer that are NOT supported by the context.")
    groundedness_score: float = Field(..., description="A score from 0.0 (not grounded) to 1.0 (fully grounded).")

class ConfidenceEngine:
    """
    Calculates a two-phase confidence score:
    1. Context Confidence: Evaluates the quality of the retrieved documents.
    2. Groundedness Confidence: Verifies if the generated answer is supported by the context.
    """
    def __init__(self, llm_wrapper: OllamaClient):
        self.llm_wrapper = llm_wrapper
        self.groundedness_prompt_template = """You are a meticulous fact-checker. Your task is to verify if the provided 'Generated Answer' is FULLY supported by the 'Supporting Context'.

        **Supporting Context:**
        ---
        {context}
        ---

        **Generated Answer:**
        ---
        {answer}
        ---

        Analyze the answer and determine if every claim made is directly supported by the text in the supporting context.
        Respond with a JSON object containing three fields:
        - "is_grounded": A boolean (true/false).
        - "unsupported_claims": A list of strings, where each string is a direct quote of a claim from the answer that is NOT supported by the context. If all claims are supported, return an empty list.
        - "groundedness_score": A float between 0.0 and 1.0, where 1.0 means fully supported and 0.0 means not supported at all.

        Respond with ONLY the JSON object.
        """

    def calculate_context_confidence(self, query_metadata: Dict[str, Any], retrieved_docs: List[Dict], reranked_docs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Calculates a confidence score based on the quality of the retrieved context."""
        # ... (previous logic for relevance, reranker, coverage, consistency)
        # This logic remains the same as before.
        docs_for_retrieval_score = retrieved_docs
        docs_for_rerank_score = reranked_docs

        if not docs_for_retrieval_score and not docs_for_rerank_score:
            return self._format_result(0, "low", "No documents found", {})

        vector_scores = [doc.get('score', 100) for doc in docs_for_retrieval_score if doc.get('score') is not None]
        relevance_scores = [np.exp(-0.1 * s) for s in vector_scores]
        relevance_score_avg = np.mean(relevance_scores) if relevance_scores else 0.0

        reranker_scores = [doc.get('score', 0.0) for doc in docs_for_rerank_score if doc.get('score') is not None] if docs_for_rerank_score else []
        reranker_score_avg = np.mean(reranker_scores) if reranker_scores else 0.0
        
        num_docs = len(reranked_docs) if reranked_docs is not None else len(retrieved_docs)
        coverage_score = min(num_docs / 5.0, 1.0)

        score_std_dev = np.std(reranker_scores if reranker_scores else relevance_scores)
        consistency_score = 1 - min(score_std_dev, 1.0)

        if reranker_scores:
            weights = {"reranker": 0.6, "relevance": 0.1, "coverage": 0.2, "consistency": 0.1}
            context_score = (weights["reranker"] * reranker_score_avg +
                             weights["relevance"] * relevance_score_avg +
                             weights["coverage"] * coverage_score +
                             weights["consistency"] * consistency_score)
        else:
            weights = {"relevance": 0.7, "coverage": 0.2, "consistency": 0.1}
            context_score = (weights["relevance"] * relevance_score_avg +
                             weights["coverage"] * coverage_score +
                             weights["consistency"] * consistency_score)

        context_score = max(0, min(1, context_score))

        details = {
            "relevance_score_avg": round(relevance_score_avg, 4),
            "reranker_score_avg": round(reranker_score_avg, 4),
            "coverage_score": round(coverage_score, 4),
            "consistency_score": round(consistency_score, 4),
            "num_docs": num_docs,
            "weights": weights
        }
        
        return {"context_score": context_score, "details": details}

    async def verify_groundedness(self, answer: str, context_docs: List[Dict]) -> GroundednessCheck:
        """Uses an LLM to verify if the answer is grounded in the provided context."""
        context_str = "\n\n".join([doc.get('content', '') for doc in context_docs])
        prompt = self.groundedness_prompt_template.format(context=context_str, answer=answer)
        
        try:
            response_text = await self.llm_wrapper.generate_from_prompt(prompt, model=settings.LARGE_LLM_MODEL)
            # Basic JSON extraction
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                json_content = response_text[start:end+1]
                return GroundednessCheck.parse_raw(json_content)
            else:
                raise ValueError("No JSON object found in groundedness check response.")
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to verify groundedness: {e}")
            return GroundednessCheck(is_grounded=False, unsupported_claims=[answer], groundedness_score=0.0)

    def get_final_confidence(self, context_confidence: Dict, groundedness_check: GroundednessCheck, query_metadata: Dict) -> Dict:
        """Combines context and groundedness scores into a final confidence assessment."""
        context_score = context_confidence.get("context_score", 0.0)
        groundedness_score = groundedness_check.groundedness_score

        # The final score is heavily weighted towards groundedness
        final_score = (0.3 * context_score) + (0.7 * groundedness_score)
        
        # Dynamic thresholds based on query complexity
        complexity = query_metadata.get("complexity", "medium")
        thresholds = {
            "low": {"high": 0.75, "medium": 0.60},
            "medium": {"high": 0.85, "medium": 0.70},
            "high": {"high": 0.90, "medium": 0.75}
        }
        
        current_thresholds = thresholds.get(complexity, thresholds["medium"])

        if final_score >= current_thresholds["high"]:
            level = "high"
            reason = "High context quality and strong answer groundedness."
        elif final_score >= current_thresholds["medium"]:
            level = "medium"
            reason = "Acceptable context and groundedness, but could be improved."
        else:
            level = "low"
            reason = "Low context quality or significant parts of the answer are not supported by the context."

        details = {
            **context_confidence.get("details", {}),
            "groundedness_score": groundedness_score,
            "unsupported_claims": groundedness_check.unsupported_claims,
            "final_score_weights": {"context": 0.3, "groundedness": 0.7}
        }

        return self._format_result(final_score, level, reason, details)

    def decide_action_from_context(self, context_confidence: Dict, is_post_expansion: bool = False) -> str:
        """Decides an action based only on the initial context confidence."""
        score = context_confidence.get("context_score", 0.0)
        
        # Use simpler thresholds for this initial decision
        if score >= 0.7:
            return "generate"
        elif score >= 0.5 and not is_post_expansion:
            return "expand"
        elif is_post_expansion: # If score is still low after expansion, generate anyway and hope for the best
            return "generate"
        else:
            return "expand"

    def _format_result(self, score: float, level: str, reason: str, details: Dict) -> Dict[str, Any]:
        return {
            "confidence_score": round(score, 4),
            "confidence_level": level,
            "reason": reason,
            "details": details
        }