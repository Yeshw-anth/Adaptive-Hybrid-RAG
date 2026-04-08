
import pandas as pd
from typing import Dict, Any
import logging

def process_table(element: Any) -> Dict[str, Any]:
    """
    Processes a table element from unstructured, handling potential parsing errors.

    Args:
        element (Any): A table element from unstructured.

    Returns:
        Dict[str, Any]: A dictionary containing the table as markdown and as structured JSON.
                        Returns empty strings if parsing fails.
    """
    try:
        table_html = element.metadata.text_as_html
        logging.info(f"Attempting to parse table HTML: {table_html}")
        if not table_html:
            logging.warning("Skipping table element with no HTML representation.")
            return {"table_text": "", "table_data": "{}"}

        df = pd.read_html(str(table_html), header=0)[0]

        table_markdown = df.to_markdown(index=False)
        table_json = df.to_json(orient="split")

        return {
            "table_text": table_markdown,
            "table_data": table_json
        }
    except Exception as e:
        logging.error(f"Failed to process table element: {e}")
        # Return empty data to prevent crashing the pipeline
        return {
            "table_text": "",
            "table_data": "{}"
        }