import os
import json
import math
import time
from datetime import datetime
from typing import List, Dict, Any, Generator

import pandas as pd
from datasets import Dataset
from loguru import logger

from ragas import evaluate, config
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings

# ============================================================
# Production Configuration
# ============================================================

MAX_RETRIES = 3
MAX_ANSWER_LENGTH = 1024


# ============================================================
# Google Gemini LLM Configuration
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=settings.GOOGLE_MODEL_NAME,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True
)

ragas_llm = LangchainLLMWrapper(llm)


# ============================================================
# Embedding Configuration
# ============================================================

ragas_embeddings = HuggingFaceEmbeddings(
    model=settings.EMBED_MODEL_NAME,
)


# ============================================================
# Metrics Configuration
# ============================================================

faithfulness_metric = Faithfulness(
    llm=ragas_llm
)

answer_relevancy_metric = AnswerRelevancy(
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

context_precision_metric = ContextPrecision(
    llm=ragas_llm
)

context_recall_metric = ContextRecall(
    llm=ragas_llm
)

METRICS = [
    faithfulness_metric,
    answer_relevancy_metric,
    context_precision_metric,
    context_recall_metric,
]


# ============================================================
# Utility Helpers
# ============================================================

def clean_nan_values(data):
    """
    Recursively replace NaN values with None
    so JSON serialization remains valid.
    """

    if isinstance(data, dict):
        return {
            k: clean_nan_values(v)
            for k, v in data.items()
        }

    elif isinstance(data, list):
        return [
            clean_nan_values(v)
            for v in data
        ]

    elif isinstance(data, float) and math.isnan(data):
        return None

    return data


# ============================================================
# Main Evaluator
# ============================================================

class RagasEvaluator:
    """
    Production-safe RAGAS evaluator for fully local
    RAG evaluation pipelines using:

    - Ollama
    - LangChain wrappers
    - HuggingFace embeddings

    Features:
    - fully local inference
    - batching
    - retry logic
    - safe JSONL persistence
    - robust error handling
    - incremental scoring
    """

    def __init__(
        self,
        log_file_path: str = None,
        batch_size: int = 1,
        ground_truth_file_path: str = None,
    ):
        self.log_file_path = (
            log_file_path
            or settings.OUTPUT_LOG_FILE
        )

        self.batch_size = batch_size
        self.ground_truth_file_path = ground_truth_file_path

        logger.info(
            f"RagasEvaluator initialized "
            f"(batch_size={self.batch_size})"
        )

        logger.info(
            f"LLM: {settings.DEFAULT_LLM_MODEL}"
        )

        logger.info(
            f"Embeddings: {settings.EMBED_MODEL_NAME}"
        )

    # ========================================================
    # Load Logs
    # ========================================================

    def _load_and_filter_logs(
        self
    ) -> List[Dict[str, Any]]:
        """
        Load only unevaluated logs.
        """

        if not os.path.exists(self.log_file_path):
            logger.warning(
                "Log file not found."
            )
            return []

        unevaluated_logs = []

        with open(
            self.log_file_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                try:
                    log = json.loads(line)

                    if any(
                        metric.name not in log
                        for metric in METRICS
                    ):
                        unevaluated_logs.append(log)

                except (
                    json.JSONDecodeError,
                    KeyError,
                ) as e:

                    logger.warning(
                        f"Skipping malformed log: {e}"
                    )

        return unevaluated_logs

    # ========================================================
    # Batch Generator
    # ========================================================

    def _batch_generator(
        self,
        logs: List[Dict[str, Any]],
    ) -> Generator[List[Dict[str, Any]], None, None]:

        for i in range(
            0,
            len(logs),
            self.batch_size
        ):
            yield logs[i:i + self.batch_size]

    # ========================================================
    # Update Logs
    # ========================================================

    def _update_logs_in_place(
        self,
        scored_logs: List[Dict[str, Any]],
    ):
        """
        Safely merge evaluated scores
        back into the JSONL file.
        """

        temp_file_path = (
            str(self.log_file_path) + ".tmp"
        )

        scored_logs_map = {
            (
                log["timestamp"],
                log["query"]
            ): log
            for log in scored_logs
        }

        try:

            with open(
                self.log_file_path,
                "r",
                encoding="utf-8"
            ) as infile, open(
                temp_file_path,
                "w",
                encoding="utf-8"
            ) as outfile:

                for line in infile:

                    original_log = json.loads(line)

                    key = (
                        original_log.get("timestamp"),
                        original_log.get("query"),
                    )

                    if key in scored_logs_map:

                        cleaned_log = clean_nan_values(
                            scored_logs_map[key]
                        )

                        outfile.write(
                            json.dumps(cleaned_log)
                            + "\n"
                        )

                    else:
                        outfile.write(line)

            os.replace(
                temp_file_path,
                self.log_file_path
            )

            logger.info(
                "Successfully updated log file."
            )

        except Exception as e:

            logger.error(
                f"Failed updating logs: {e}",
                exc_info=True,
            )

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # ========================================================
    # Main Evaluation
    # ========================================================

    def _load_ground_truth(self) -> Dict[str, Dict[str, Any]]:
        """Loads the ground truth dataset from the predefined path."""
        gt_path = self.ground_truth_file_path or settings.EVALUATION_DATASET_PATH
        if not os.path.exists(gt_path):
            logger.warning(f"Ground truth file not found at {gt_path}. Context metrics will be skipped.")
            return {}

        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth_data = json.load(f)

        # Create a map from query to ground truth for easy lookup
        return {item['query']: item for item in ground_truth_data}

    def run_evaluation(self) -> Dict[str, Any]:
        ground_truth_map = self._load_ground_truth()
        unevaluated_logs = self._load_and_filter_logs()

        if not unevaluated_logs:
            logger.info("No logs to evaluate.")
            return {
                "message": "All logs are already evaluated.",
                "evaluated_count": 0,
                "average_scores": {},
            }

        logger.info(f"Found {len(unevaluated_logs)} unevaluated logs.")

        all_scored_logs = []
        total_evaluated_count = 0

        for batch_logs in self._batch_generator(unevaluated_logs):
            logger.info(f"Processing batch ({len(batch_logs)} logs)")

            evaluation_data = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": [],
            }
            valid_batch_logs = []

            for log in batch_logs:
                query = log.get("query")
                ground_truth_item = ground_truth_map.get(query)

                if not ground_truth_item:
                    logger.warning(f"Skipping log for query '{query}' as no ground truth was found.")
                    continue

                answer = log.get("final_answer", "")[:MAX_ANSWER_LENGTH]
                contexts = [doc.get("text") for doc in log.get("final_docs", []) if doc.get("text")]

                if query and answer and contexts:
                    evaluation_data["question"].append(query)
                    evaluation_data["answer"].append(answer)
                    evaluation_data["contexts"].append(contexts)
                    # RAGAS expects 'ground_truth' to be a single string, so we join if it's a list.
                    gt = ground_truth_item.get("ground_truth_answer", "")
                    if isinstance(gt, list):
                        gt = "\n".join(gt)
                    evaluation_data["ground_truth"].append(gt)
                    
                    valid_batch_logs.append(log)
                else:
                    logger.warning("Skipping invalid log entry due to missing query, answer, or contexts.")

            if not valid_batch_logs:
                logger.warning("No valid logs in batch to evaluate.")
                continue

            dataset = Dataset.from_dict(evaluation_data)
            
            result = None
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Running RAGAS (attempt {attempt + 1})")
                    result = evaluate(
                        dataset,
                        metrics=METRICS,
                        raise_exceptions=True,
                    )
                    break
                except Exception as e:
                    logger.warning(f"Evaluation failed on attempt {attempt + 1}: {e}")
                    time.sleep(2)

            if result is None:
                logger.error("Batch failed after all retries.")
                continue
        
            try:
                scores_df = result.to_pandas()
                scores_dict = scores_df.to_dict("records")

                for i, log in enumerate(valid_batch_logs):
                    if i < len(scores_dict):
                        log.update(scores_dict[i])
                        log["evaluation_timestamp"] = datetime.now().isoformat()
                        log["evaluation_model"] = settings.DEFAULT_LLM_MODEL
                        log["embedding_model"] = settings.EMBED_MODEL_NAME
                
                all_scored_logs.extend(valid_batch_logs)
                total_evaluated_count += len(valid_batch_logs)
            
            except Exception as e:
                logger.error(f"Failed to process scores: {e}", exc_info=True)

        if not all_scored_logs:
            return {
                "message": "Evaluation ran, but no new logs were scored.",
                "evaluated_count": 0,
                "average_scores": {},
            }

        self._update_logs_in_place(all_scored_logs)

        # Calculate and return average scores
        avg_scores = self._calculate_average_scores(all_scored_logs)
        logger.info(f"Successfully evaluated and updated {total_evaluated_count} log entries.")

        return {
            "message": f"Evaluation complete. Processed {total_evaluated_count} logs.",
            "evaluated_count": total_evaluated_count,
            "average_scores": avg_scores,
        }

    def _calculate_average_scores(self, scored_logs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates the average score for each metric."""
        scores = {metric.name: [] for metric in METRICS}
        for log in scored_logs:
            for metric in METRICS:
                if metric.name in log and log[metric.name] is not None:
                    scores[metric.name].append(log[metric.name])
        
        avg_scores = {
            metric: sum(values) / len(values) if values else 0
            for metric, values in scores.items()
        }
        logger.info(f"Average scores: {avg_scores}")
        return avg_scores

if __name__ == "__main__":
    import pprint

    print("Starting RAGAS evaluation...")
    evaluator = RagasEvaluator()
    evaluation_results = evaluator.run_evaluation()
    print("\n--- RAGAS Evaluation Complete ---")
    pprint.pprint(evaluation_results)
    print("---------------------------------")