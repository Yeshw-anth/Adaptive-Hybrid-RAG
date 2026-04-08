from typing import List, Dict, Any
import pandas as pd
from io import StringIO
from llama_index.core.schema import TextNode


from src.chunking.strategies.base_strategy import BaseChunkingStrategy
from src.data.schemas import DocumentMetadata

class TableChunkingStrategy(BaseChunkingStrategy):
    """
    A strategy for chunking tables.

    This strategy converts a table into two formats:
    1.  A Markdown representation for embedding and semantic search.
    2.  A JSON representation for structured metadata.

    It creates a single chunk for the entire table.
    """
    async def process(self, elements: List[Any], file_path: str) -> List[TextNode]:
        """
        Processes a table section by creating a single, comprehensive parent node
        that includes both Markdown and JSON representations of the table.
        No child nodes are created for tables.
        """
        table_html = "\n\n".join([el.metadata.text_as_html for el in elements if hasattr(el.metadata, 'text_as_html')]).strip()
        if not table_html:
            return []

        try:
            df = pd.read_html(StringIO(table_html))[0]
            markdown_table = df.to_markdown(index=False)
            json_table = df.to_json(orient='records')

            node = TextNode(
                text=markdown_table,
                metadata={
                    "structured_data_json": json_table,
                    "has_structure": True,
                    "content_type": "table",
                }
            )
            return [node]

        except Exception as e:
            # Fallback for parsing errors
            return []