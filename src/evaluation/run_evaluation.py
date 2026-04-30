from src.core.logging_config import logger
import os
import asyncio
from typing import List, Dict, Any

from src.core.system_builder import SystemBuilder
from src.data.schemas import Document
from ragevaluator import RAGEvaluator

# --- Logging Setup ---
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "evaluation.log")

# Configure Loguru for file and console logging
logger.remove() # Remove default handler to control all outputs
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
    level="INFO"
)
logger.add(
    log_file_path,
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
    level="INFO",
    mode="w"
)
# --- End Logging Setup ---


async def main():
    """
    Runs an evaluation of the RAG system against a predefined dataset.
    This script uses the same SystemBuilder as the main application to ensure
    we are evaluating the real, production-configured pipeline.
    """
    logger.info("--- Starting RAG System Evaluation ---")

    # 1. Build the RAG system using the centralized builder
    logger.info("Building RAG system from SystemBuilder...")
    try:
        builder = SystemBuilder()
        # We only need the pipeline, not the index object itself for evaluation
        rag_pipeline, _ = builder.build_all()
        logger.info("RAG system built successfully.")
    except Exception as e:
        logger.error(f"Failed to build the RAG system: {e}", exc_info=True)
        return

    # 2. Define evaluation configurations
    # These configurations will be passed to the pipeline to test different strategies.
    # Note: Our new architecture uses strategies, not simple boolean flags.
    # We will simulate these by overriding parts of the query analysis.
    configs = [
        {
            "name": "Fast (Vector Retrieval Only)",
            "strategy_override": {"pipeline": "fast", "use_reranker": False, "use_expansion": False}
        },
        {
            "name": "Accurate (Hybrid + Rerank)",
            "strategy_override": {"pipeline": "accurate", "use_reranker": True, "use_expansion": False}
        },
        {
            "name": "Accurate + Expansion",
            "strategy_override": {"pipeline": "accurate", "use_reranker": True, "use_expansion": True}
        }
    ]

    # 3. Define the evaluation dataset path
    dataset_path = os.path.join(str(builder.components['embedder'].model.get_sentence_embedding_dimension()), "evaluation_dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Evaluation dataset not found at: {dataset_path}")
        logger.error("Please ensure you have a dataset file (e.g., evaluation_dataset.json) in the project root.")
        return

    # 4. Run evaluation for different models
    models_to_evaluate = ["phi3", "llama3"]
    all_results = []

    for model_name in models_to_evaluate:
        logger.info(f"\n--- Evaluating Model: {model_name} ---")
        
        evaluator = RAGEvaluator(
            pipeline=rag_pipeline,
            model_name=model_name,
            configs=configs
        )
        
        try:
            results_df = await evaluator.evaluate(dataset_path)
            evaluator.generate_report(results_df, model_name)
            all_results.append(results_df)
        except Exception as e:
            logger.error(f"An error occurred during evaluation for model {model_name}: {e}", exc_info=True)

    logger.info("--- RAG System Evaluation Finished ---")
    # Here you could add logic to combine or compare reports if needed.


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"A critical error occurred in the main execution block: {e}", exc_info=True)
    finally:
        pass  # Loguru handles shutdown automatically