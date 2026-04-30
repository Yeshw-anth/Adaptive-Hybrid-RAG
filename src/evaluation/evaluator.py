from src.core.logging_config import logger
from typing import List, Dict, Any
from src.core.llm.ollama_client import OllamaClient


class Evaluator:
    """
    A class to evaluate the performance of the RAG system using metrics like
    retrieval recall and answer faithfulness.
    """
    def __init__(self, llm_client: OllamaClient):
        """
        Initializes the evaluator with an LLM client to be used for LLM-as-judge evaluations.
        """
        self.llm_client = llm_client

    def _calculate_retrieval_recall(self, initial_docs: List[Dict], final_docs: List[Dict]) -> float:
        """
        Calculates a proxy for retrieval recall.

        This metric checks what percentage of the documents used to generate the final answer
        were present in the initial retrieval set. A high score means the initial retrieval
        was effective.

        Args:
            initial_docs: The list of documents from the very first retrieval step.
            final_docs: The list of documents that were ultimately fed to the LLM to generate the answer.

        Returns:
            A recall score between 0.0 and 1.0.
        """
        if not final_docs:
            return 1.0 # If no docs were needed, recall is perfect.
        if not initial_docs and final_docs:
            return 0.0 # If docs were needed but initial retrieval found none.

        initial_doc_ids = {doc["metadata"]["chunk_id"] for doc in initial_docs}
        final_doc_ids = {doc["metadata"]["chunk_id"] for doc in final_docs}

        retrieved_and_used_ids = initial_doc_ids.intersection(final_doc_ids)

        recall = len(retrieved_and_used_ids) / len(final_doc_ids)
        return recall

    def _evaluate_faithfulness(self, answer: str, context_docs: List[Dict]) -> Dict[str, Any]:
        """
        Uses an LLM-as-a-judge to evaluate the factual faithfulness of an answer
        to its provided context.

        Args:
            answer: The generated answer from the RAG pipeline.
            context_docs: The source documents provided to the LLM to generate the answer.

        Returns:
            A dictionary containing the faithfulness score (1-5) and the judge's reasoning.
        """
        if not answer or not context_docs:
            return {"score": 1, "reasoning": "Faithfulness not evaluated due to missing answer or context."}

        context_str = "\n".join([doc["text"] for doc in context_docs])
        
        prompt = f"""Please act as an impartial judge. Your task is to evaluate if the following `ANSWER` is factually supported by the provided `CONTEXT`.

        **CONTEXT:**
        {context_str}

        **ANSWER:**
        {answer}

        **INSTRUCTIONS:**
        1. Compare the `ANSWER` to the `CONTEXT` and determine if all claims in the answer can be verified by the context.
        2. Provide a score from 1 to 5, where 1 is "not at all supported" and 5 is "fully supported".
        3. Provide a brief `reasoning` for your score.
        4. Output your response in a JSON format with two keys: "score" and "reasoning".

        **EXAMPLE OUTPUT:**
        {{
          "score": 5,
          "reasoning": "The answer correctly states the capital of France is Paris, which is directly supported by the context."
        }}
        """
        
        try:
            response = self.llm_client.generate_response(prompt, json_mode=True)
            # The response from the LLM should be a JSON string, so we parse it.
            evaluation_result = self.llm_client.parse_json(response)
            if "score" not in evaluation_result or "reasoning" not in evaluation_result:
                raise ValueError("LLM judge response did not contain 'score' and 'reasoning' keys.")
            return evaluation_result
        except Exception as e:
            logger.error(f"LLM-as-judge evaluation failed: {e}")
            return {"score": 0, "reasoning": f"Evaluation failed due to an error: {e}"}

    def evaluate(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a full evaluation on the output of a RAG pipeline.

        Args:
            pipeline_result: The dictionary returned by a pipeline's `run` method.

        Returns:
            A dictionary containing all evaluation metrics.
        """
        eval_results = {}

        # 1. Retrieval Recall
        initial_docs = pipeline_result.get("retrieved_docs", [])
        final_docs = pipeline_result.get("final_docs_for_generation", [])
        eval_results["retrieval_recall"] = self._calculate_retrieval_recall(initial_docs, final_docs)

        # 2. Faithfulness
        answer = pipeline_result.get("answer")
        eval_results["faithfulness"] = self._evaluate_faithfulness(answer, final_docs)

        logger.info(f"Evaluation complete: {eval_results}")
        return eval_results