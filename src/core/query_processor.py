import re
import string
from typing import List

# It's better to use a pre-existing, well-maintained list of stop words.
# If NLTK is not a project dependency, we can define a minimal list.
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except (ImportError, LookupError):
    print("NLTK 'stopwords' resource not found or NLTK is not installed. Using a basic list of stop words.")
    # A small, common list of English stop words.
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
        'will', 'with', 'i', 'you', 'your', 'we', 'our', 'they', 'their'
    }

class QueryProcessor:
    """
    A class to handle various forms of query preprocessing for different retrieval strategies.
    """

    def __init__(self, stop_words: set = None):
        """
        Initializes the QueryProcessor.

        Args:
            stop_words (set, optional): A set of stop words to use. 
                                       Defaults to a standard English set.
        """
        self.stop_words = stop_words or STOP_WORDS

    def normalize(self, query: str) -> str:
        """
        Performs basic normalization suitable for semantic search.
        - Converts to lowercase.
        - Removes extra whitespace.

        Args:
            query (str): The input user query.

        Returns:
            str: The normalized query.
        """
        if not isinstance(query, str):
            return ""
        # Collapse whitespace and strip leading/trailing spaces
        query = re.sub(r'\s+', ' ', query).strip()
        return query.lower()

    def clean_for_keywords(self, query: str) -> List[str]:
        """
        Performs aggressive cleaning to extract keywords for lexical search (like BM25).
        - Normalizes the query (lowercase, whitespace).
        - Removes punctuation.
        - Removes stop words.
        - Splits into tokens.

        Args:
            query (str): The input user query.

        Returns:
            List[str]: A list of cleaned keyword tokens.
        """
        # Start with basic normalization
        normalized_query = self.normalize(query)

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        no_punct_query = normalized_query.translate(translator)

        # Tokenize and remove stop words
        tokens = no_punct_query.split()
        keywords = [token for token in tokens if token not in self.stop_words]

        return keywords

if __name__ == '__main__':
    # Example Usage
    processor = QueryProcessor()
    
    raw_query = "  What is the  best way to learn about Auto-Adaptive RAG systems? "
    
    # --- For Semantic Search ---
    semantic_query = processor.normalize(raw_query)
    print(f"Original Query: '{raw_query}'")
    print(f"Semantic Query: '{semantic_query}'")
    # Expected output: 'what is the best way to learn about auto-adaptive rag systems?'

    print("-" * 20)

    # --- For Keyword Search ---
    keyword_tokens = processor.clean_for_keywords(raw_query)
    print(f"Original Query: '{raw_query}'")
    print(f"Keyword Tokens: {keyword_tokens}")
    # Expected output: ['what', 'best', 'way', 'learn', 'autoadaptive', 'rag', 'systems']