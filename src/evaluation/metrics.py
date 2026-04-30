from src.core.logging_config import logger
from typing import List
from sentence_transformers import util

def precision_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """
    Calculates precision at k.
    """
    if not retrieved_sources:
        return 0.0
    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)
    return len(retrieved_set.intersection(expected_set)) / len(retrieved_set)

def recall_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """
    Calculates recall at k.
    """
    if not expected_sources:
        return 0.0
    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)
    return len(retrieved_set.intersection(expected_set)) / len(expected_set)

def recall_at_k(retrieved_ids: List[str], expected_id: str, k: int) -> int:
    """
    Checks if the expected document ID is within the top-k retrieved documents.
    """
    return 1 if expected_id in retrieved_ids[:k] else 0

def mean_reciprocal_rank(retrieved_ids: List[str], expected_id: str) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR) for a single query.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id == expected_id:
            return 1.0 / (i + 1)
    return 0.0

def semantic_similarity(generated_answer: str, ground_truth_answer: str, model) -> float:
    """
    Computes the cosine similarity between the generated answer and the ground truth answer.
    """
    embeddings = model.encode([generated_answer, ground_truth_answer])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def faithfulness_score(generated_answer: str, context: str, model) -> float:
    """
    Computes the semantic similarity between the generated answer and the retrieved context.
    """
    embeddings = model.encode([generated_answer, context])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def keyword_overlap(answer: str, expected_keywords: List[str]) -> float:
    """
    Calculates the fraction of expected keywords found in the answer.
    """
    if not expected_keywords:
        return 0.0
    found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
    return found_keywords / len(expected_keywords)

def hallucination_flag(answer: str, confidence: float, threshold: float = 0.1) -> bool:
    """
    Flags potential hallucinations based on low confidence and lack of expected keywords.
    """
    if confidence < threshold and not any(keyword.lower() in answer.lower() for keyword in ["not available", "don't know"]):
        return True
    return False

def hallucination_check(answer: str, context: str, model) -> bool:
    """
    Checks for hallucinations using context-grounded answer check and answer length sanity rule.
    """
    if not context and len(answer) > 20:
        return True

    if not context:
        return False

    embeddings = model.encode([answer, context])
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    if similarity < 0.65:
        return True
    
    return False