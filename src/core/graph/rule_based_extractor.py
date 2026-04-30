# src/core/graph/rule_based_extractor.py

import spacy
from spacy.matcher import Matcher
from typing import List, Tuple
from src.core.logging_config import logger

class RuleBasedTripleExtractor:
    """
    Extracts triples using spaCy's dependency parsing and custom matching rules.
    """
    def __init__(self):
        logger.info("Initializing RuleBasedTripleExtractor...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.matcher = self._setup_matcher()
            logger.info("RuleBasedTripleExtractor initialized successfully.")
        except OSError:
            logger.error("Failed to load spaCy model 'en_core_web_sm'.")
            logger.info("Please ensure the model is installed by running: python -m spacy download en_core_web_sm")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during initialization: {e}", exc_info=True)
            raise

    def _setup_matcher(self) -> Matcher:
        """Sets up the spaCy Matcher with patterns for triple extraction."""
        matcher = Matcher(self.nlp.vocab)
        
        pattern1 = [
            {"DEP": "nsubj", "OP": "?"},
            {"DEP": "ROOT"},
            {"DEP": "attr", "OP": "?"},
            {"DEP": "dobj", "OP": "?"}
        ]
        
        pattern2 = [
            {"DEP": "nsubjpass", "OP": "?"},
            {"DEP": "ROOT"},
            {"DEP": "agent", "OP": "?"}
        ]

        matcher.add("SVO", [pattern1, pattern2])
        return matcher

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Main extraction function. It processes text and uses multiple methods
        to find triples.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        
        dep_triples = self._extract_from_dependencies(doc)
        matcher_triples = self._extract_from_matcher(doc)

        combined_triples = list(set(dep_triples + matcher_triples))
        
        logger.info(f"Extracted {len(combined_triples)} unique triples from text.")
        return combined_triples

    def _extract_from_dependencies(self, doc) -> List[Tuple[str, str, str]]:
        """Extracts SVO triples from dependency parse tree."""
        triples = []
        for token in doc:
            if "subj" in token.dep_:
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if "obj" in child.dep_:
                        obj = child.text
                        triples.append((subject, verb, obj))
        return triples

    def _extract_from_matcher(self, doc) -> List[Tuple[str, str, str]]:
        """Extracts triples using predefined matcher patterns."""
        triples = []
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if len(span) >= 3:
                triples.append((span[0].text, span[1].text, span[2].text))
        return triples