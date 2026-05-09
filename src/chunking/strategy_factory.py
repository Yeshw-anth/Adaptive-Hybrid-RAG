from typing import Dict, Type

from src.chunking.strategies.base_strategy import BaseChunkingStrategy

class StrategyFactory:
    """
    A factory for creating and providing content-specific chunking strategies.
    This allows the chunking engine to be easily extended with new strategies for
    different types of content.
    """

    def __init__(self, strategies: Dict[str, BaseChunkingStrategy]):
        self._strategies = strategies

    def get_strategy(self, content_type: str, language: str = None) -> BaseChunkingStrategy:
        """
        Returns the appropriate chunking strategy for the given content type.
        If a specific strategy is not found, it defaults to the text strategy.

        Args:
            content_type: The type of content to be chunked (e.g., "text", "table").
            language: The programming language for code content, if applicable.

        Returns:
            An instance of a BaseChunkingStrategy.
        """
        if content_type == "code":
            # The CodeChunkingStrategy can be enhanced to handle different languages
            return self._strategies.get("code", self._strategies["text"])

        return self._strategies.get(content_type, self._strategies["text"])