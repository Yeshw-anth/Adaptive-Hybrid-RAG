from src.core.logging_config import logger
from typing import List, Dict, Any
from llama_index.core.schema import TextNode
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.chunking.strategies.base_strategy import BaseChunkingStrategy
from src.data.schemas import DocumentMetadata


class MarkdownChunkingStrategy(BaseChunkingStrategy):
    """
    A chunking strategy for Markdown documents using LangChain's MarkdownTextSplitter.
    This avoids the LLM dependency of LlamaIndex's parser.
    """

    def __init__(self):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        logger.info(f"Initialized MarkdownChunkingStrategy with LangChain MarkdownHeaderTextSplitter.")

    async def process(self, elements: List[Any], file_path: str) -> List[TextNode]:
        """
        Processes a full Markdown document using the MarkdownHeaderTextSplitter.
        """
        content = "\n\n".join([el.text for el in elements]).strip()
        if not content:
            return []

        # Split the text into chunks using the LangChain splitter
        langchain_docs = self.splitter.split_text(content)

        final_nodes = []
        for i, doc in enumerate(langchain_docs):
            node = TextNode(
                text=doc.page_content,
                metadata={
                    "content_type": "markdown"
                }
            )
            final_nodes.append(node)

        logger.info(f"Processed Markdown document '{file_path}' into {len(final_nodes)} nodes using MarkdownTextSplitter.")
        return final_nodes