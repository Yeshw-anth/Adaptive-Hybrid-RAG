import pytest
from unittest.mock import MagicMock, patch
from src.core.strategy.router import StrategyRouter
from src.data.schemas import QueryMetadata, Strategy
from src.core.decision.cost_latency_controller import CostLatencyController
from src.config.settings import settings

@pytest.fixture
def mock_cost_latency_controller():
    """Provides a mock CostLatencyController."""
    controller = MagicMock(spec=CostLatencyController)
    # Default to always being within budget unless specified otherwise
    controller.is_within_budget.return_value = True
    return controller

@pytest.fixture
def strategy_router(mock_cost_latency_controller):
    """Provides a StrategyRouter instance with a mocked controller."""
    return StrategyRouter(mock_cost_latency_controller)

def test_select_strategy_simple_query(strategy_router):
    """
    Tests that a simple query gets the 'fast' pipeline by default.
    """
    metadata = QueryMetadata(
        intent="fact-seeking",
        complexity="low",
        query_type="simple",
        normalized_query="q", keyword_tokens=[], keywords=[], expected_answer_format="list"
    )
    strategy = strategy_router.select_strategy(metadata)
    assert strategy.pipeline == "fast"

def test_select_strategy_complex_query(strategy_router):
    """
    Tests that a complex query gets the 'accurate' pipeline by default.
    """
    metadata = QueryMetadata(
        intent="summary",
        complexity="high",
        query_type="complex",
        normalized_query="q", keyword_tokens=[], keywords=[], expected_answer_format="list"
    )
    strategy = strategy_router.select_strategy(metadata)
    assert strategy.pipeline == "accurate"

def test_apply_contextual_rules_high_complexity(strategy_router):
    """
    Tests that high complexity upgrades to hybrid retrieval and reranker.
    """
    metadata = QueryMetadata(
        complexity="high",
        intent="fact-seeking", query_type="complex", normalized_query="q", keyword_tokens=[], keywords=[], expected_answer_format="list"
    )
    params = strategy_router._get_default_strategy(metadata)
    modified_params = strategy_router._apply_contextual_rules(params, metadata)
    assert modified_params["retrieval_strategy"] == "hybrid"
    assert modified_params["use_reranker"] is True

def test_adjust_for_budget_upgrade(strategy_router, mock_cost_latency_controller):
    """
    Tests that the strategy is upgraded if the budget allows.
    """
    metadata = QueryMetadata(
        complexity="high",
        intent="fact-seeking", query_type="complex", normalized_query="q", keyword_tokens=[], keywords=[], expected_answer_format="list"
    )
    # The default accurate strategy starts with a small model
    strategy = strategy_router.select_strategy(metadata, model_override="test-large-model")
    # The mock controller allows all upgrades, so it should upgrade to the large model
    assert strategy.model == "test-large-model"

    
def test_adjust_for_budget_downgrade(strategy_router, mock_cost_latency_controller):
    """
    Tests that the strategy is downgraded if it exceeds the budget.
    """
    with patch.object(strategy_router, 'strategy_levels', [
        {"model": "test-large-model", "use_reranker": True, "retrieval_strategy": "hybrid", "top_k": 12},
        {"model": "test-medium-model", "use_reranker": True, "retrieval_strategy": "hybrid", "top_k": 10},
        {"model": "test-small-model", "use_reranker": False, "retrieval_strategy": "vector", "top_k": 8},
    ]):
        metadata = QueryMetadata(
            complexity="high",
            intent="fact-seeking", query_type="complex", normalized_query="q", keyword_tokens=[], keywords=[], expected_answer_format="list"
        )
        # Simulate a scenario where even the initial strategy is too expensive
        mock_cost_latency_controller.is_within_budget.side_effect = [
            False,  # The first (best) strategy is too expensive
            True,   # The second strategy fits the budget
        ]

        # The router will select the initial strategy, apply rules, and then adjust for budget.
        # The mock will force it to downgrade to the second-best option.
        strategy = strategy_router.select_strategy(metadata)

        # It should downgrade to the medium model and keep the reranker
        assert strategy.model == "test-medium-model"
        assert strategy.use_reranker

def test_get_default_accurate_strategy(strategy_router):
    """
    Tests the fallback to a default accurate strategy.
    """
    strategy = strategy_router.get_default_accurate_strategy()
    assert strategy.pipeline == "accurate"
    assert strategy.retrieval_strategy == "hybrid"
    assert strategy.use_reranker is True
    assert strategy.top_k == 10