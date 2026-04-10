import logging
from typing import List, Any
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from langchain_core.documents import Document
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
        Processes a list of unstructured elements by converting them to LangChain
        Documents and then splitting them.
        """
        if not elements:
            return []

        # Convert unstructured elements to LangChain Documents.
        # This preserves the boundaries between elements.
        langchain_docs = [Document(page_content=el.text) for el in elements if hasattr(el, 'text') and el.text.strip()]

        if not langchain_docs:
            logger.warning("No processable text found in the provided elements.")
            return []

        # Use the splitter to create chunks from the list of Documents.
        text_chunks = self.splitter.split_documents(langchain_docs)

        all_nodes = []
        for i, chunk in enumerate(text_chunks):
            logger.debug(
                f"Creating node {i+1}/{len(text_chunks)}. "
                f"Content: '{chunk.page_content[:80].strip()}...'"
            )
            # Convert the LangChain Document chunk back to a LlamaIndex TextNode
            node = TextNode(text=chunk.page_content)
            all_nodes.append(node)
        
        return all_nodes