import logging
import uuid
from typing import List, Dict, Any
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import CodeSplitter
from tree_sitter import Language, Parser

from src.chunking.strategies.base_strategy import BaseChunkingStrategy
from src.data.schemas import DocumentMetadata
from src.config import settings

logger = logging.getLogger(__name__)

# Path to the compiled language library is now managed in settings
LANGUAGE_LIBRARY_PATH = str(settings.LANGUAGE_LIBRARY_PATH)

class CodeChunkingStrategy(BaseChunkingStrategy):
    """
    A chunking strategy for source code.
    Uses a CodeSplitter to break down code into logical units like functions or classes.
    """

    def __init__(self, language: str = "python", max_chars: int = 1500):
        try:
            logger.info("Attempting to initialize CodeChunkingStrategy...")
            # Create a parser and set its language
            parser = Parser()
            logger.info(f"Loading language '{language}' from library: {LANGUAGE_LIBRARY_PATH}")
            language_obj = Language(LANGUAGE_LIBRARY_PATH, language)
            logger.info("Language loaded successfully.")
            parser.set_language(language_obj)
            logger.info("Parser language set successfully.")
            
            self.splitter = CodeSplitter(language=language, max_chars=max_chars, parser=parser)
            logger.info(f"Initialized CodeChunkingStrategy for language: {language}")
        except Exception as e:
            logger.error(f"Failed to initialize CodeChunkingStrategy for {language}: {e}")
            # Fallback or raise exception
            raise ValueError(f"Unsupported language for CodeChunkingStrategy: {language}") from e

    async def process(self, elements: List[Any], file_path: str) -> List[TextNode]:
        """
        Processes a code section by creating a parent node for the entire section
        and then splitting it into smaller, language-aware child nodes.
        """
        parent_text = "\n\n".join([el.text for el in elements]).strip()
        if not parent_text:
            return []

        # 1. Create the parent node for the entire code section
        parent_node = TextNode(
            text=parent_text,
            metadata={
                "is_parent": True,
                "content_type": "code"
            }
        )
        logger.info(f"Created PARENT node for code section.")

        # 2. Create child nodes by splitting the parent text
        child_nodes = []
        chunks = self.splitter.split_text(parent_text)
        for i, chunk_text in enumerate(chunks):
            logger.debug(f"Creating CODE CHILD node {i+1}/{len(chunks)}.")

            child_node = TextNode(
                text=f"""```{self.splitter.language}\n{chunk_text}\n```""",
                metadata={
                    "is_parent": False,
                    "content_type": "code"
                }
            )
            child_nodes.append(child_node)
        
        return [parent_node] + child_nodes