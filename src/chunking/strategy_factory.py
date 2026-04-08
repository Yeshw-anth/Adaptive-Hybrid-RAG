from typing import Dict, Type

from src.chunking.strategies.base_strategy import BaseChunkingStrategy
from src.chunking.strategies.text_strategy import TextChunkingStrategy
from src.chunking.strategies.table_strategy import TableChunkingStrategy
from src.chunking.strategies.code_strategy import CodeChunkingStrategy
from src.chunking.strategies.markdown_strategy import MarkdownChunkingStrategy
from src.chunking.strategies.image_strategy import ImageChunkingStrategy
from llama_index.core.embeddings import BaseEmbedding

class StrategyFactory:
    """
    A factory for creating and providing content-specific chunking strategies.
    This allows the chunking engine to be easily extended with new strategies for
    different types of content.
    """

    def __init__(self, embedding_model: BaseEmbedding):
        self.embedding_model = embedding_model
        self._strategies: Dict[str, BaseChunkingStrategy] = {
            "text": TextChunkingStrategy(embedding_model=self.embedding_model),
            "table": TableChunkingStrategy(),
            "code": CodeChunkingStrategy(),
            "markdown": MarkdownChunkingStrategy(),
            "image": ImageChunkingStrategy(),
        }

    def get_strategy(self, content_type: str, language: str = None) -> BaseChunkingStrategy:
        """
        Returns the appropriate chunking strategy for the given content type.
        If a specific strategy is not found, it defaults to the text strategy.

        Args:
            content_type: The type of content to be chunked (e.g., "text", "table").

        Returns:
            An instance of a BaseChunkingStrategy.
        """
        if content_type == "code" and language:
            return CodeChunkingStrategy(language=language)
        
        # Return a new instance of the strategy if it needs the embedding model
        if content_type in ["text", "markdown"]:
            return self._strategies[content_type]
            
        return self._strategies.get(content_type, self._strategies["text"])