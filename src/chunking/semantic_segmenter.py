import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from unstructured.documents.elements import Element
import nltk

# Ensure the 'punkt' tokenizer data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class SemanticSegmenter:
    """
    Segments a document based on semantic similarity of sentences.
    This segmenter identifies logical breaks in the narrative flow.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.4):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        logger.info(f"SemanticSegmenter initialized with model '{model_name}' and threshold {similarity_threshold}")

    def segment_document(self, elements: List[Element]) -> List[Dict[str, Any]]:
        """
        Performs semantic segmentation on a list of document elements.

        Instead of returning text, this now returns the original elements, grouped
        into semantic sections.

        Returns:
            A list of dictionaries, where each dictionary represents a semantic section
            with a 'title' and a list of 'elements'.
        """
        if not elements:
            logger.warning("No elements provided for semantic segmentation.")
            return []

        # Create a list of sentences linked to their parent element
        sentence_element_pairs: List[Tuple[str, Element]] = []
        for el in elements:
            if hasattr(el, 'text') and el.text.strip():
                sentences = nltk.sent_tokenize(el.text)
                for sentence in sentences:
                    sentence_element_pairs.append((sentence, el))
        
        if not sentence_element_pairs:
            logger.warning("No sentences could be extracted from the elements.")
            return []

        sentences = [pair[0] for pair in sentence_element_pairs]

        if len(sentences) < 2:
            logger.debug("Not enough sentences to compare. Returning all elements as a single section.")
            return [{"title": "Full Document", "elements": elements}]

        try:
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
            similarities = self._calculate_cosine_similarities(embeddings)
            
            breakpoints = np.where(similarities < self.similarity_threshold)[0]
            
            sections = []
            last_breakpoint = 0
            for i, breakpoint_index in enumerate(breakpoints):
                # Get the elements corresponding to the sentences in this section
                start_el_index = last_breakpoint
                end_el_index = breakpoint_index + 1
                
                # Collect all unique elements from this sentence range, preserving order
                elements_slice = [pair[1] for pair in sentence_element_pairs[start_el_index:end_el_index]]
                seen_ids = set()
                unique_elements_in_order = []
                for el in elements_slice:
                    if id(el) not in seen_ids:
                        seen_ids.add(id(el))
                        unique_elements_in_order.append(el)
                section_elements = sorted(unique_elements_in_order, key=lambda e: elements.index(e))

                if section_elements:
                    sections.append({
                        "title": f"Semantic Section {i + 1}",
                        "elements": section_elements
                    })
                last_breakpoint = breakpoint_index + 1
            
            # Add the last remaining section
            elements_slice = [pair[1] for pair in sentence_element_pairs[last_breakpoint:]]
            seen_ids = set()
            unique_elements_in_order = []
            for el in elements_slice:
                if id(el) not in seen_ids:
                    seen_ids.add(id(el))
                    unique_elements_in_order.append(el)
            final_elements = sorted(unique_elements_in_order, key=lambda e: elements.index(e))

            if final_elements:
                sections.append({
                    "title": f"Semantic Section {len(sections) + 1}",
                    "elements": final_elements
                })

            logger.info(f"Successfully created {len(sections)} semantic sections containing original elements.")
            return self._merge_consecutive_identical_sections(sections)

        except Exception as e:
            logger.error(f"Failed to perform semantic segmentation: {e}", exc_info=True)
            # Fallback to returning the whole document as one section on error
            return [{"title": "Full Document", "elements": elements}]

    def _calculate_cosine_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = embeddings.cpu().numpy()
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
            similarities.append(sim)
        return np.array(similarities)

    def _merge_consecutive_identical_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges sections that have the exact same set of elements."""
        if not sections:
            return []
        
        merged_sections = [sections[0]]
        for i in range(1, len(sections)):
            # Using element IDs for a more robust comparison
            current_element_ids = {id(el) for el in sections[i]['elements']}
            prev_element_ids = {id(el) for el in merged_sections[-1]['elements']}
            
            if current_element_ids == prev_element_ids:
                # This section is identical to the previous one, so skip it
                continue
            else:
                merged_sections.append(sections[i])
        
        # Re-title the sections after merging
        for i, section in enumerate(merged_sections):
            section['title'] = f"Semantic Section {i + 1}"
            
        return merged_sections