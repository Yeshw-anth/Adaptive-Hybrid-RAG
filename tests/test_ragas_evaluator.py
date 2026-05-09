import json
import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.evaluation.ragas_evaluator import RagasEvaluator


# -----------------------------
# Test Data
# -----------------------------

UNEVALUATED_LOG = {
    "query": "What is online SSL?",
    "final_answer": "It's a machine learning technique.",
    "final_docs": [
        {
            "text": "Online SSL allows models to learn from streaming data."
        }
    ],
    "timestamp": "2023-10-27T10:00:00Z",
}

EVALUATED_LOG = {
    "query": "What is adaptive learning?",
    "final_answer": "It's a system that adapts.",
    "final_docs": [
        {
            "text": "Adaptive systems change behavior dynamically."
        }
    ],
    "timestamp": "2023-10-27T11:00:00Z",
    "faithfulness": 0.95,
    "answer_relevancy": 0.91,
}


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def mock_ground_truth_file(tmp_path):
    """
    Creates a temporary ground truth JSON file.
    """
    gt_data = [
        {
            "query": "What is online SSL?",
            "ground_truth_answer": "Online Semi-Supervised Learning (SSL) is a machine learning approach where a model learns from a small amount of labeled data and a large amount of unlabeled data in a streaming or online setting. This allows the model to adapt to new data patterns over time without requiring constant manual labeling."
        }
    ]
    gt_file = tmp_path / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt_data, f)
    return str(gt_file)


@pytest.fixture
def mock_log_file(tmp_path):
    """
    Creates temporary JSONL log file.
    """
    log_file = tmp_path / "test_logs.jsonl"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(UNEVALUATED_LOG) + "\n")
        f.write(json.dumps(EVALUATED_LOG) + "\n")

    return str(log_file)


# -----------------------------
# Happy Path Test
# -----------------------------

@patch("src.evaluation.ragas_evaluator.evaluate")
def test_run_evaluation_happy_path(
    mock_evaluate,
    mock_log_file,
    mock_ground_truth_file
):
    """
    Tests successful batch evaluation flow.
    """

    # Mock ragas response
    mock_result = MagicMock()

    mock_result.to_pandas.return_value = pd.DataFrame({
        "faithfulness": [1.0],
        "answer_relevancy": [0.92],
        "context_precision": [0.89],
        "context_recall": [0.95],
    })

    mock_evaluate.return_value = mock_result

    evaluator = RagasEvaluator(
        log_file_path=mock_log_file,
        batch_size=1,
        ground_truth_file_path=mock_ground_truth_file,
    )

    result = evaluator.run_evaluation()

    # -----------------------------
    # Assertions
    # -----------------------------

    # evaluate() called once
    mock_evaluate.assert_called_once()

    # verify dataset content
    dataset_arg = mock_evaluate.call_args[0][0]

    assert dataset_arg["question"][0] == UNEVALUATED_LOG["query"]

    # verify result
    assert result["evaluated_count"] == 1

    assert (
        result["message"]
        == "Evaluation complete. Processed 1 logs."
    )

    # verify file updated
    with open(mock_log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2

    updated_log = json.loads(lines[0])

    assert updated_log["faithfulness"] == 1.0
    assert updated_log["answer_relevancy"] == 0.92
    assert updated_log["context_precision"] == 0.89
    assert updated_log["context_recall"] == 0.95

    untouched_log = json.loads(lines[1])

    assert untouched_log["faithfulness"] == 0.95


# -----------------------------
# Missing File Test
# -----------------------------

def test_run_evaluation_no_log_file():
    """
    Tests missing log file handling.
    """

    missing_file = "/tmp/does_not_exist.jsonl"

    if os.path.exists(missing_file):
        os.remove(missing_file)

    evaluator = RagasEvaluator(
        log_file_path=missing_file
    )

    result = evaluator.run_evaluation()

    assert result["evaluated_count"] == 0

    assert (
        result["message"]
        == "All logs are already evaluated."
        or "No logs found" in result["message"]
    )


# -----------------------------
# Empty Batch Test
# -----------------------------

@patch("src.evaluation.ragas_evaluator.evaluate")
def test_empty_batch_skipped(
    mock_evaluate,
    tmp_path
):
    """
    Tests logs with missing contexts are skipped.
    """

    bad_log = {
        "query": "bad query",
        "final_answer": "bad answer",
        "final_docs": [],
        "timestamp": "123",
    }

    log_file = tmp_path / "bad_logs.jsonl"

    with open(log_file, "w") as f:
        f.write(json.dumps(bad_log) + "\n")

    evaluator = RagasEvaluator(
        log_file_path=str(log_file)
    )

    result = evaluator.run_evaluation()

    mock_evaluate.assert_not_called()

    assert result["evaluated_count"] == 0


# -----------------------------
# Ragas Failure Recovery Test
# -----------------------------

@patch("src.evaluation.ragas_evaluator.evaluate")
def test_ragas_failure_recovery(
    mock_evaluate,
    mock_log_file,
    mock_ground_truth_file
):
    """
    Ensures evaluator survives Ragas crashes.
    """

    mock_evaluate.side_effect = Exception(
        "Ollama timeout"
    )

    evaluator = RagasEvaluator(
        log_file_path=mock_log_file,
        ground_truth_file_path=mock_ground_truth_file
    )

    result = evaluator.run_evaluation()

    assert result["evaluated_count"] == 0

    assert "no new logs were scored" in result["message"].lower()