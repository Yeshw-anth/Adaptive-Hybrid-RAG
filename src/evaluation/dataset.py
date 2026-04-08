import json
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_test_dataset(path: str) -> List[Dict]:
    """
    Loads a test dataset from a JSON file and validates its structure.

    Args:
        path (str): The path to the JSON file.

    Returns:
        List[Dict]: A list of validated query objects.
    """
    try:
        with open(path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        logging.error(f"Dataset file not found at: {path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Malformed JSON in dataset file: {path}")
        return []

    validated_dataset = []
    for i, item in enumerate(dataset):
        if not all(k in item for k in ["query", "expected_keywords", "expected_sources"]):
            logging.warning(f"Skipping malformed entry at index {i} in {path}: Missing required keys.")
            continue
        validated_dataset.append(item)
        
    return validated_dataset

if __name__ == '__main__':
    # Example usage:
    test_dataset_path = "test_queries.json"
    test_queries = load_test_dataset(test_dataset_path)
    
    if test_queries:
        print(f"Successfully loaded {len(test_queries)} validated queries.")
        print("First query:", test_queries[0])