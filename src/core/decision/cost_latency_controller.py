from typing import Dict, Any

class CostLatencyController:
    """
    Estimates the operational cost and latency for a given RAG strategy
    and validates it against user-defined constraints.
    """
    def __init__(self):
        """
        Initializes the controller with predefined estimates for various components.
        - Latency is in seconds.
        - Cost is a normalized value (e.g., USD per 1,000 queries).
        """
        self.pipeline_estimates = {
            "fast": {"latency": 0.5, "cost": 0.01},
            "accurate": {"latency": 2.5, "cost": 0.10},
            "code": {"latency": 2.0, "cost": 0.08},
            "structured": {"latency": 3.0, "cost": 0.12},
        }
        self.model_estimates = {
            "small_llm": {"latency": 0.8, "cost": 0.02},
            "large_llm": {"latency": 3.0, "cost": 0.20},
        }
        self.feature_estimates = {
            "use_reranker": {"latency": 0.5, "cost": 0.005},
            "use_hybrid_retrieval": {"latency": 0.2, "cost": 0.002},
            "expand_query": {"latency": 1.5, "cost": 0.05}  # Expansion involves an extra LLM call
        }

    def _estimate_strategy_cost(self, strategy: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimates the total cost and latency for a given strategy dictionary.
        """
        pipeline_name = strategy.get("pipeline", "fast")
        model = strategy.get("model", "small_llm")

        # Start with pipeline base cost
        total_latency = self.pipeline_estimates.get(pipeline_name, {}).get("latency", 0)
        total_cost = self.pipeline_estimates.get(pipeline_name, {}).get("cost", 0)

        # Add model cost
        total_latency += self.model_estimates.get(model, {}).get("latency", 0)
        total_cost += self.model_estimates.get(model, {}).get("cost", 0)

        # Add feature costs
        if strategy.get("use_reranker"):
            total_latency += self.feature_estimates["use_reranker"]["latency"]
            total_cost += self.feature_estimates["use_reranker"]["cost"]

        if strategy.get("use_hybrid_retrieval"):
            total_latency += self.feature_estimates["use_hybrid_retrieval"]["latency"]
            total_cost += self.feature_estimates["use_hybrid_retrieval"]["cost"]
        
        # Note: Expansion cost is not pre-estimated as it's a dynamic, second-step action.
        # The initial strategy selection must be within budget.

        return {"estimated_latency": total_latency, "estimated_cost": total_cost}

    def is_within_budget(self, strategy: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        """
        Checks if the estimated cost and latency of a strategy are within the
        constraints defined in the query metadata.
        """
        max_latency = query_metadata.get("max_latency")
        max_cost = query_metadata.get("max_cost")

        # If no constraints are set, the strategy is always valid.
        if max_latency is None and max_cost is None:
            return True

        estimates = self._estimate_strategy_cost(strategy)

        if max_latency is not None and estimates["estimated_latency"] > max_latency:
            print(f"Strategy rejected: Estimated latency {estimates['estimated_latency']:.2f}s > max_latency {max_latency}s")
            return False
        
        if max_cost is not None and estimates["estimated_cost"] > max_cost:
            print(f"Strategy rejected: Estimated cost ${estimates['estimated_cost']:.4f} > max_cost ${max_cost:.4f}")
            return False
            
        return True