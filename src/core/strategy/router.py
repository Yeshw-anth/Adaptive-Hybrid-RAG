from src.core.logging_config import logger
from src.config import settings
from src.data.schemas import QueryMetadata, Strategy
from src.core.decision.cost_latency_controller import CostLatencyController


class StrategyRouter:
    """
    Selects and fine-tunes the optimal RAG strategy based on query metadata,
    cost, and latency constraints, allowing for both upgrades and downgrades.
    """
    def __init__(self, cost_latency_controller: CostLatencyController):
        self.cost_latency_controller = cost_latency_controller

    def select_strategy(self, query_metadata: QueryMetadata, model_override: str = None) -> Strategy:
        """
        Determines the best strategy by starting with a balanced default, applying
        rules, and then adjusting for cost/latency budget.
        """
        logger.info("--- Strategy Selection & Tuning Process Started ---")
        logger.info(f"Received Query Metadata: {query_metadata.model_dump_json(indent=2)}")

        # 1. Start with a balanced, default strategy
        strategy_params = self._get_default_strategy(query_metadata)
        logger.info(f"Initial balanced strategy: {strategy_params}")

        # 2. Apply contextual rules to modify the strategy
        strategy_params = self._apply_contextual_rules(strategy_params, query_metadata)
        logger.info(f"Strategy after applying rules: {strategy_params}")

        # 3. Adjust strategy based on cost/latency budget (Upgrade or Downgrade)
        final_strategy = self._adjust_for_budget(strategy_params, query_metadata, model_override)

        logger.info(f"Final Tuned Strategy: {final_strategy.model_dump_json(indent=2)}")
        return final_strategy

    def get_default_accurate_strategy(self) -> Strategy:
        """
        Returns a hard-coded, sensible default 'accurate' strategy.
        This is used as a reliable fallback when the analysis process fails.
        """
        logger.warning("Using hard-coded default 'accurate' strategy as a fallback.")
        return Strategy(
            pipeline="accurate",
            retrieval_strategy="hybrid",
            use_reranker=True,
            model=settings.DEFAULT_LLM_MODEL,
            top_k=10
        )


    def _get_default_strategy(self, query_metadata: QueryMetadata) -> dict:
        """Returns a sensible default strategy based on initial query analysis."""
        # Use the "fast" pipeline only for the simplest, most direct queries.
        is_truly_simple = (
            query_metadata.complexity == "low" and 
            query_metadata.intent == "fact-seeking" and 
            query_metadata.query_type == "simple"
        )

        if is_truly_simple:
            logger.info("Query assessed as simple. Starting with 'fast' pipeline.")
            return {
                "pipeline": "fast",
                "retrieval_strategy": "vector",
                "use_reranker": False,
                "model": settings.SMALL_LLM_MODEL,
                "top_k": 5
            }
        
        logger.info("Query requires deeper analysis. Starting with 'accurate' pipeline.")
        return {
            "pipeline": "accurate",
            "retrieval_strategy": "vector", # Start with vector, can be upgraded
            "use_reranker": False,          # Start without reranker, can be upgraded
            "model": settings.SMALL_LLM_MODEL, # Start with small model, can be upgraded
            "top_k": 8
        }

    def _apply_contextual_rules(self, params: dict, metadata: QueryMetadata) -> dict:
        """Applies a set of rules to modify the strategy based on query metadata."""
        # Rule: Prioritize keyword pipeline for specific query types
        if metadata.query_type == "keyword":
            logger.info("Rule Applied: Query type is 'keyword' -> Switching to 'keyword' pipeline.")
            params["pipeline"] = "keyword"
            params["retrieval_strategy"] = "keyword" # Explicitly set for clarity
            params["use_reranker"] = False
            params["model"] = settings.SMALL_LLM_MODEL # Keyword searches are often simpler
            return params # Exit early as this is a pipeline switch

        # Rule: Switch to a specialized pipeline if content hints suggest it.
        # This rule takes precedence.
        if "table" in metadata.content_hints:
            logger.info("Rule Applied: Content hint 'table' -> Switching to 'structured' pipeline.")
            params["pipeline"] = "structured"
            params["use_reranker"] = False # Rerankers often not needed for structured data
            return params # Exit early as this is a pipeline switch

        if "code" in metadata.content_hints or any(k in metadata.keywords for k in ["python", "javascript", "function"]):
             logger.info("Rule Applied: Content hint 'code' -> Switching to 'code' pipeline.")
             params["pipeline"] = "code"
             return params # Exit early

        # Rule: High complexity or analytical intent suggests deeper, more accurate retrieval
        if metadata.complexity == "high" or metadata.intent == "causal-analysis":
            logger.info("Rule Applied: High complexity/analysis -> Upgrading to hybrid retrieval and reranker.")
            params["retrieval_strategy"] = "hybrid"
            params["use_reranker"] = True
            params["top_k"] = max(params.get("top_k", 8), 12)

        # Rule: Summarization requires more context
        if metadata.intent == "summary":
            logger.info("Rule Applied: Summary intent -> Increasing retrieval depth.")
            params["top_k"] = max(params.get("top_k", 8), 15)

        # Rule: Comparisons benefit from a reranker and more documents
        if metadata.intent == "comparison":
            logger.info("Rule Applied: Comparison intent -> Enabling reranker and increasing depth.")
            params["use_reranker"] = True
            params["top_k"] = max(params.get("top_k", 8), 10)
            
        return params

    def _adjust_for_budget(self, strategy_params: dict, query_metadata: QueryMetadata, model_override: str = None) -> Strategy:
        """
        Checks a strategy against the budget and iteratively adjusts it.
        It can upgrade features if there is budget, or downgrade if it's too expensive.
        """
        query_dict = query_metadata.model_dump()

        if model_override:
            logger.info(f"Model override is active. Locking model to {model_override}.")
            strategy_params["model"] = model_override

        # --- Attempt to UPGRADE if budget allows ---
        upgrade_steps = [
            ("model", settings.LARGE_LLM_MODEL),
            ("use_reranker", True),
            ("retrieval_strategy", "hybrid"),
        ]
        
        current_strategy = strategy_params.copy()
        logger.info("Checking for potential upgrades...")
        for feature, upgraded_value in upgrade_steps:
            if feature == "model" and model_override:
                logger.info("Skipping model upgrade due to override.")
                continue
            if current_strategy.get(feature) != upgraded_value:
                potential_upgrade = current_strategy.copy()
                potential_upgrade[feature] = upgraded_value
                if self.cost_latency_controller.is_within_budget(potential_upgrade, query_dict):
                    logger.info(f"Upgrade successful: Setting '{feature}' to '{upgraded_value}'.")
                    current_strategy = potential_upgrade
                else:
                    logger.info(f"Cannot afford to upgrade '{feature}'.")

        # If the (potentially upgraded) strategy is within budget, we are done.
        if self.cost_latency_controller.is_within_budget(current_strategy, query_dict):
            logger.info("Final strategy is within budget.")
            return Strategy(**current_strategy)

        # --- If not, DOWNGRADE from the last working (or initial) config ---
        logger.warning("Strategy exceeds budget. Attempting to downgrade.")
        
        downgrade_steps = [
            ("model", settings.SMALL_LLM_MODEL),
            ("use_reranker", False),
            ("retrieval_strategy", "vector"),
        ]

        for feature, downgraded_value in downgrade_steps:
            if feature == "model" and model_override:
                logger.info("Skipping model downgrade due to override.")
                continue
            if current_strategy.get(feature) != downgraded_value:
                logger.info(f"Downgrade Step: Setting '{feature}' to '{downgraded_value}'.")
                current_strategy[feature] = downgraded_value
                if self.cost_latency_controller.is_within_budget(current_strategy, query_dict):
                    logger.info("Downgraded strategy now fits within budget.")
                    return Strategy(**current_strategy)
        
        logger.error("Could not find a strategy that fits the budget, even after all downgrades.")
        # Return the most-downgraded version as a last resort
        return Strategy(**current_strategy)