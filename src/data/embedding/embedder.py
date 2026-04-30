from src.core.logging_config import logger
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Any

class Embedder(HuggingFaceEmbedding):
    """
    A specialized HuggingFace embedding model for our RAG system.

    This class inherits from LlamaIndex's HuggingFaceEmbedding to ensure
    it's fully compatible with the LlamaIndex ecosystem (e.g., VectorStoreIndex).
    It automatically handles model loading, tokenization, and embedding generation.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs: Any):
        """
        Initializes the specialized Embedder.

        Args:
            model_name (str): The name of the sentence-transformer model to use.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        logger.info(f"Initializing specialized Embedder with model: {model_name}")
        super().__init__(model_name=model_name, **kwargs)
        logger.info("Specialized Embedder initialized successfully.")

# The old methods (embed_chunks, embed_query) are no longer needed
# as the parent HuggingFaceEmbedding class handles the core embedding logic
# in a way that's compatible with LlamaIndex.