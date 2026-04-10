import logging
from typing import List, Dict, Any, Literal
from src.chunking.structure_segmenter import StructuralSegmenter
from src.chunking.semantic_segmenter import SemanticSegmenter

logger = logging.getLogger(__name__)

class SegmentationEvaluator:
    """
    Orchestrates document segmentation using a hierarchical strategy.
    It delegates to specialized segmenters (structural and semantic)
    and implements a fallback mechanism.
    """
    def __init__(self, structural_segmenter: StructuralSegmenter, semantic_segmenter: SemanticSegmenter):
        self.structural_segmenter = structural_segmenter
        self.semantic_segmenter = semantic_segmenter

    def segment_document(
        self, 
        elements: List[Any], 
        file_path: str, 
        initial_strategy: Literal['structural', 'semantic'] = 'structural'
    ) -> (List[Dict[str, Any]], str):
        """
        Segments the document using a hierarchical fallback strategy with strict validation.
        Returns the sections and the name of the strategy that was successful.
        """
        logger.info(f"Starting segmentation for '{file_path}' with initial strategy: '{initial_strategy}'")
        
        if not elements:
            return [], "no_elements"

        sections = []
        successful_strategy = ""

        # 1. Attempt structural segmentation. Success is defined as > 1 section.
        if initial_strategy == 'structural':
            structural_sections = self._run_structural_segmentation(elements, file_path)
            if len(structural_sections) > 1:
                logger.info(f"Structural segmentation successful for '{file_path}'.")
                sections = structural_sections
                successful_strategy = 'structural'

        # 2. If structural failed, attempt semantic segmentation with strict validation.
        if not successful_strategy:
            logger.warning(f"Structural segmentation was ineffective. Falling back to semantic segmentation.")
            semantic_sections = self._run_semantic_segmentation(elements, file_path)
            if self._is_semantic_segmentation_valid(semantic_sections):
                logger.info(f"Semantic segmentation successful for '{file_path}' after passing strict validation.")
                sections = semantic_sections
                successful_strategy = 'semantic'
            else:
                logger.warning(f"Semantic segmentation for '{file_path}' failed strict validation.")

        # 3. If all strategies fail, fallback to a single section containing all elements.
        if not successful_strategy:
            logger.warning("All primary segmentation strategies were ineffective. Falling back to a single document chunk.")
            successful_strategy = "full_document_fallback"
            if elements:
                sections = [{"title": "Full Document (Fallback)", "elements": elements}]

        if not sections:
            logger.error(f"All segmentation attempts for {file_path} failed. No sections generated.")
            return [], "all_strategies_failed"
        
        logger.info(f"Successfully segmented '{file_path}' using strategy: '{successful_strategy}'.")
        return sections, successful_strategy

    def _run_structural_segmentation(self, elements: List[Any], file_path: str) -> List[Dict[str, Any]]:
        """Delegates structural segmentation to the StructuralSegmenter."""
        logger.debug(f"Delegating structural segmentation for '{file_path}' to StructuralSegmenter.")
        try:
            return self.structural_segmenter.segment_document(elements)
        except Exception as e:
            logger.error(f"An error occurred during structural segmentation delegation for '{file_path}': {e}", exc_info=True)
            return []

    def _run_semantic_segmentation(self, elements: List[Any], file_path: str) -> List[Dict[str, Any]]:
        """Delegates semantic segmentation to the SemanticSegmenter."""
        logger.debug(f"Delegating semantic segmentation for '{file_path}' to SemanticSegmenter.")
        try:
            return self.semantic_segmenter.segment_document(elements)
        except Exception as e:
            logger.error(f"An error occurred during semantic segmentation delegation for '{file_path}': {e}", exc_info=True)
            return []

    def _is_semantic_segmentation_valid(self, sections: List[Dict[str, Any]], min_char_threshold: int = 100, max_tiny_fraction: float = 0.5) -> bool:
        """
        Performs a strict check to see if the semantic segmentation is valid.
        A valid segmentation must have more than one section and not have an
        excessive number of very short sections.
        """
        if not sections or len(sections) <= 1:
            logger.debug("Semantic segmentation is invalid: resulted in 1 or 0 sections.")
            return False

        section_lengths = [len("\n".join(el.text for el in section.get('elements', []))) for section in sections]
        num_tiny_sections = sum(1 for length in section_lengths if length < min_char_threshold)

        fraction_of_tiny_sections = num_tiny_sections / len(sections)

        if fraction_of_tiny_sections > max_tiny_fraction:
            logger.warning(f"Semantic segmentation is invalid: {fraction_of_tiny_sections:.1%} of sections are below {min_char_threshold} characters.")
            return False
        
        logger.debug("Semantic segmentation passed strict validation.")
        return True