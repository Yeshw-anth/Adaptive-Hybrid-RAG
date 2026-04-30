from src.core.logging_config import logger
import json
import pandas as pd
from typing import List, Dict, Any
import asyncio
import os

from src.config import settings

from src.core.orchestrator import RAGPipeline
from src.evaluation.evaluator import Evaluator


class RAGEvaluator:
    """
    Orchestrates the evaluation of a RAG pipeline against a dataset for various configurations.
    """
    def __init__(self, pipeline: RAGPipeline, model_name: str, configs: List[Dict[str, Any]]):
        self.pipeline = pipeline
        self.model_name = model_name
        self.configs = configs
        # The 'judge' evaluator for metrics like faithfulness
        self.judge = Evaluator(llm_client=pipeline.query_analyzer.llm_wrapper)

    async def _evaluate_single_case(self, question: str, ground_truth_answer: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single question-answer pair for a given configuration.
        """
        logger.info(f"Evaluating question: '{question}' with config: '{config['name']}'")
        
        # Override the model for this run
        self.pipeline.model_router.select_model = lambda qm, s: self.model_name
        
        # This is a simplified way to force a strategy for evaluation purposes.
        # In a real scenario, you might need a more robust way to override strategy selection.
        if 'strategy_override' in config:
            # Temporarily override the strategy router's selection
            original_select_strategy = self.pipeline.strategy_router.select_strategy
            
            def mock_select_strategy(query_metadata):
                from src.data.schemas import Strategy
                strategy_data = config['strategy_override']
                return Strategy(
                    pipeline=strategy_data.get('pipeline', 'fast'),
                    use_reranker=strategy_data.get('use_reranker', False),
                    use_expansion=strategy_data.get('use_expansion', False),
                    model=self.model_name
                )
            
            self.pipeline.strategy_router.select_strategy = mock_select_strategy

        try:
            # Execute the pipeline
            result = await self.pipeline.orchestrate_query(question)
            
            # Restore original strategy selection method
            if 'strategy_override' in config:
                self.pipeline.strategy_router.select_strategy = original_select_strategy

            # Use the judge to get faithfulness and other metrics
            judge_eval = self.judge.evaluate(result)

            return {
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "generated_answer": result.get("answer", "N/A"),
                "config_name": config["name"],
                "model_name": self.model_name,
                "latency": result.get("latency", 0),
                "retrieval_recall": judge_eval.get("retrieval_recall", 0),
                "faithfulness_score": judge_eval.get("faithfulness", {}).get("score", 0),
                "faithfulness_reasoning": judge_eval.get("faithfulness", {}).get("reasoning", ""),
                "context": result.get("context", "")
            }
        except Exception as e:
            logger.error(f"Error evaluating case: {e}", exc_info=True)
            # Restore original strategy selection method in case of error
            if 'strategy_override' in config and 'original_select_strategy' in locals():
                self.pipeline.strategy_router.select_strategy = original_select_strategy
            return {
                "question": question,
                "ground_truth_answer": ground_truth_answer,
                "generated_answer": f"ERROR: {e}",
                "config_name": config["name"],
                "model_name": self.model_name,
            }

    async def evaluate(self, dataset_path: str) -> pd.DataFrame:
        """
        Runs the evaluation across the entire dataset for all configurations.
        """
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse dataset from {dataset_path}: {e}")
            return pd.DataFrame()

        all_results = []
        for item in dataset:
            for config in self.configs:
                result = await self._evaluate_single_case(item["question"], item["answer"], config)
                all_results.append(result)
        
        return pd.DataFrame(all_results)

    def generate_report(self, results_df: pd.DataFrame, model_name: str):
        """Generates and saves a CSV report from the evaluation results."""
        if results_df.empty:
            logger.warning("Cannot generate report from empty results.")
            return

        report_dir = settings.EVAL_REPORT_DIR
        os.makedirs(report_dir, exist_ok=True)
        
        # Sanitize model name for filename
        safe_model_name = model_name.replace("/", "_")
        
        report_path = os.path.join(report_dir, f"evaluation_report_{safe_model_name}.csv")
        results_df.to_csv(report_path, index=False)
        
        logger.info(f"Evaluation report for model '{model_name}' saved to {report_path}")
        
        # Also print a summary to the console
        summary = results_df.groupby('config_name').agg({
            'retrieval_recall': 'mean',
            'faithfulness_score': 'mean',
            'latency': 'mean'
        }).reset_index()
        
        logger.info(f"\n--- Evaluation Summary for {model_name} ---")
        logger.info(summary.to_string(index=False))
        logger.info("-------------------------------------------------")