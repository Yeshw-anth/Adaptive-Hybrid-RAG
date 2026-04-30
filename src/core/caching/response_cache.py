# src/core/caching/response_cache.py
from src.core.logging_config import logger
from typing import List, Optional, Dict, Any
from diskcache import Cache
import hashlib

from src.config.settings import settings

class ResponseCache:
    """
    A persistent, on-disk cache for storing and retrieving RAG responses.

    This cache is designed to be "context-aware." It generates a key not just
    from the query text, but also from the specific document chunks used as
    context. This ensures that if the underlying data changes, the cache
    will not return a stale answer.
    """

    def __init__(self, cache_dir: str = settings.CACHE_DIR, ttl: int = 3600 * 24 * 7):
        """
        Initializes the response cache.

        Args:
            cache_dir (str): The directory to store the cache files.
            ttl (int): The time-to-live for cache entries, in seconds. Defaults to 7 days.
        """
        self.cache = Cache(cache_dir)
        self.ttl = ttl
        logger.info(f"ResponseCache initialized at: {cache_dir}")

    def _generate_key(self, query: str, context_node_ids: List[str]) -> str:
        """
        Generates a stable, unique key for a given query and its context.

        The key is a SHA-256 hash of the query combined with a sorted,
        concatenated string of the context node IDs.

        Args:
            query (str): The user's query.
            context_node_ids (List[str]): A list of the unique IDs of the
                                          document nodes used as context.

        Returns:
            str: A unique hexadecimal key.
        """
        # Sort the node IDs to ensure the key is stable regardless of retrieval order
        sorted_ids = sorted(context_node_ids)
        
        key_string = f"{query.strip().lower()}|{'|'.join(sorted_ids)}"
        
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

    def get(self, query: str, context_node_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Retrieves a response from the cache.

        Args:
            query (str): The user's query.
            context_node_ids (List[str]): The list of context node IDs.

        Returns:
            Optional[Dict[str, Any]]: The cached response dictionary, or None if not found.
        """
        key = self._generate_key(query, context_node_ids)
        cached_response = self.cache.get(key)
        
        if cached_response:
            logger.info(f"Cache HIT for query: '{query[:50]}...'")
            return cached_response
        
        logger.info(f"Cache MISS for query: '{query[:50]}...'")
        return None

    def set(self, query: str, context_node_ids: List[str], response: Dict[str, Any]):
        """
        Stores a response in the cache.

        Args:
            query (str): The user's query.
            context_node_ids (List[str]): The list of context node IDs.
            response (Dict[str, Any]): The response dictionary to cache.
        """
        key = self._generate_key(query, context_node_ids)
        self.cache.set(key, response, expire=self.ttl)
        logger.info(f"Cached new response for query: '{query[:50]}...'")

    def clear(self):
        """Clears the entire cache."""
        self.cache.clear()
        logger.info("ResponseCache cleared.")