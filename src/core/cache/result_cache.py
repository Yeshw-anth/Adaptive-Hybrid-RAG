import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ResultCache:
    """
    A simple in-memory cache for storing RAG pipeline results.
    The cache uses a composite key of (index_version, query) to ensure
    that cached results are not stale after data ingestion.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized in-memory ResultCache.")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an item from the cache.

        Args:
            key: The composite key for the cached item.

        Returns:
            The cached result dictionary, or None if not found.
        """
        result = self._cache.get(key)
        if result:
            logger.info(f"Cache HIT for key: '{key[:50]}...'")
        else:
            logger.info(f"Cache MISS for key: '{key[:50]}...'")
        return result

    def set(self, key: str, value: Dict[str, Any]):
        """
        Adds an item to the cache.

        Args:
            key: The composite key for the cached item.
            value: The result dictionary to cache.
        """
        self._cache[key] = value
        logger.info(f"Cached result for key: '{key[:50]}...'")

    def clear(self):
        """Clears the entire cache."""
        self._cache.clear()
        logger.info("ResultCache cleared.")

    @property
    def size(self) -> int:
        """Returns the number of items in the cache."""
        return len(self._cache)