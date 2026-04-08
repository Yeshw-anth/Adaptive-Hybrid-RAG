import logging
from typing import List, Any
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.strategies.base_strategy import BaseChunkingStrategy
from src.config import settings

logger = logging.getLogger(__name__)

class TextChunkingStrategy(BaseChunkingStrategy):
    """
    A chunking strategy for plain text that uses a hierarchical, semantic-aware splitter.
    """

    def __init__(self, embedding_model: BaseEmbedding):
        super().__init__(embedding_model)
        
        # Use LangChain's splitter for more intelligent, semantic chunking.
        # It prioritizes keeping paragraphs and sentences whole.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(f"Initialized TextChunkingStrategy with RecursiveCharacterTextSplitter (Chunk Size: {settings.CHUNK_SIZE}).")

    async def process(self, elements: List[Any], file_path: str) -> List[TextNode]:
        """
        Processes a list of text elements by combining them and splitting hierarchically.
        """
        if not elements:
            return []

        # Combine the text from all elements into a single string.
        full_text = "\n\n".join([el.text for el in elements]).strip()
        if not full_text:
            return []

        # Use the hierarchical splitter to create more meaningful chunks.
        text_chunks = self.splitter.split_text(full_text)

        all_nodes = []
        for i, text_chunk in enumerate(text_chunks):
            logger.debug(
                f"Creating node {i+1}/{len(text_chunks)}. "
                f"Content: '{text_chunk[:80].strip()}...'"
            )
            node = TextNode(text=text_chunk)
            all_nodes.append(node)
        
        return all_nodes