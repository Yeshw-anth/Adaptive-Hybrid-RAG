import os
from tree_sitter import Language, Parser
from src.core.logging_config import logger
from typing import List, Any
from llama_index.core.schema import TextNode

from src.chunking.strategies.base_strategy import BaseChunkingStrategy

def get_node_text(node, text_bytes):
    """Helper to get the text content of a tree-sitter node."""
    return text_bytes[node.start_byte:node.end_byte].decode('utf-8')

class CodeChunkingStrategy(BaseChunkingStrategy):
    """
    A chunking strategy for source code that uses a custom-built tree-sitter parser.
    This strategy avoids the llama-index CodeSplitter to prevent dependency conflicts.
    """

    def __init__(self, language: str = "python", max_chars: int = 1500):
        super().__init__()
        try:
            logger.info(f"Attempting to initialize CodeChunkingStrategy for '{language}'...")
            
            # 1. Determine the path to the custom-built parser library.
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            library_path = os.path.join(project_root, "treesitter_build", "languages.dll" if os.name == 'nt' else "languages.so")

            if not os.path.exists(library_path):
                raise RuntimeError(
                    f"Tree-sitter library not found at '{library_path}'. "
                    "Ensure it was built successfully during application startup."
                )

            # 2. Load the language grammar from the compiled library.
            language_grammar = Language(library_path, language)
            
            # 3. Initialize a new parser and set its language.
            self.parser = Parser()
            self.parser.set_language(language_grammar)
            
            self.language = language
            self.max_chars = max_chars
            logger.info(f"Successfully initialized CodeChunkingStrategy with custom-built parser for '{language}'.")
        except Exception as e:
            logger.error(f"Failed to initialize CodeChunkingStrategy for '{language}': {e}", exc_info=True)
            raise ValueError(f"Unsupported or failed to load language for CodeChunkingStrategy: {language}") from e

    def _split_text(self, text: str) -> List[str]:
        """
        Splits code into chunks based on its syntax tree.
        It iterates over top-level nodes and groups them into chunks that respect the max_chars limit.
        """
        text_bytes = text.encode('utf-8')
        tree = self.parser.parse(text_bytes)
        
        if not tree.root_node or not tree.root_node.children:
            return [text] if text else []

        chunks = []
        current_chunk_nodes = []
        current_chunk_len = 0

        for node in tree.root_node.children:
            node_len = node.end_byte - node.start_byte
            if current_chunk_len > 0 and current_chunk_len + node_len > self.max_chars:
                # Finalize the current chunk
                chunk_text = get_node_text(current_chunk_nodes[0], text_bytes)
                if len(current_chunk_nodes) > 1:
                     # If there are multiple nodes, get text from start of first to end of last
                     start_byte = current_chunk_nodes[0].start_byte
                     end_byte = current_chunk_nodes[-1].end_byte
                     chunk_text = text_bytes[start_byte:end_byte].decode('utf-8')
                chunks.append(chunk_text)
                current_chunk_nodes = []
                current_chunk_len = 0

            current_chunk_nodes.append(node)
            current_chunk_len += node_len

        # Add the last remaining chunk
        if current_chunk_nodes:
            start_byte = current_chunk_nodes[0].start_byte
            end_byte = current_chunk_nodes[-1].end_byte
            chunk_text = text_bytes[start_byte:end_byte].decode('utf-8')
            chunks.append(chunk_text)
            
        return chunks

    async def process(self, elements: List[Any], file_path: str) -> List[TextNode]:
        """
        Processes a code section by creating a parent node and then splitting it into child nodes.
        """
        parent_text = "\n\n".join([el.text for el in elements]).strip()
        if not parent_text:
            return []

        parent_node = TextNode(
            text=parent_text,
            metadata={"is_parent": True, "content_type": "code"}
        )
        logger.info(f"Created PARENT node for code section.")

        child_nodes = []
        chunks = self._split_text(parent_text)
        for i, chunk_text in enumerate(chunks):
            logger.debug(f"Creating CODE CHILD node {i+1}/{len(chunks)}.")
            child_node = TextNode(
                text=f"```{self.language}\n{chunk_text}\n```",
                metadata={"is_parent": False, "content_type": "code"}
            )
            child_nodes.append(child_node)
        
        return [parent_node] + child_nodes