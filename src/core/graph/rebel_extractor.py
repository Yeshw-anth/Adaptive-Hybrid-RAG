# src/core/graph/rebel_extractor.py

import re
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.core.logging_config import logger

class RebelTripleExtractor:
    """
    Extracts knowledge triples (subject, relation, object) from text using
    the REBEL (Relation Extraction By End-to-end Language generation) model.
    This implementation avoids the transformers.pipeline to prevent auto-task
    detection errors and uses the model directly for generation.
    """
    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        """
        Initializes the extractor by loading the REBEL model and tokenizer.
        """
        logger.info(f"Initializing RebelTripleExtractor with model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info("RebelTripleExtractor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load REBEL model '{model_name}': {e}", exc_info=True)
            raise

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extracts triples from a given text by generating directly from the model.
        """
        if not text.strip():
            return []

        try:
            # Tokenize the input text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Generate token ids from the model
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    num_return_sequences=1
                )

            # Decode the generated tokens to a string
            decoded_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0]

            # Parse the generated string to extract structured triples
            triples = self._parse_triples(decoded_text)
            logger.debug(f"Extracted {len(triples)} triples using REBEL from text: '{text[:100]}...'")
            return triples
        except Exception as e:
            logger.error(f"Error during REBEL triple extraction: {e}", exc_info=True)
            return []

    def _parse_triples(self, generated_text: str) -> List[Tuple[str, str, str]]:
        """
        Parses the raw, decoded output of the REBEL model into a list of triples.
        The model outputs a string with a specific format, e.g.,
        '<s><triplet> subject <subj> relation <obj> object</s>'
        This function uses regex to robustly capture these triples.
        """
        triples = []
        # The regex looks for the <triplet> pattern and captures the text for subject, relation, and object.
        # It uses [^<]* to capture any character except '<', which prevents special tokens from being included in the content.
        pattern = r"<triplet>\s*([^<]*)\s*<subj>\s*([^<]*)\s*<obj>\s*([^<]*)"
        
        for match in re.finditer(pattern, generated_text):
            subject = match.group(1).strip()
            relation = match.group(2).strip()
            obj = match.group(3).strip()
            
            if subject and relation and obj:
                triples.append((subject, relation, obj))
        
        return list(set(triples)) # Return unique triples