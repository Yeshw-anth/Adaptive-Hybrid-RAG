from src.core.logging_config import logger
import re
from typing import List, Dict, Any
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, CompositeElement, Text
from src.data.embedding.embedder import Embedder


class StructuralSegmenter:
    """
    Performs structural segmentation based on document titles and validates the results.
    This segmenter identifies sections based on explicit titles in the document.
    """
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.irrelevant_title_patterns = [
            r"^\s*table of contents\s*$",
            r"^\s*figure\s*\d+",
            r"^\s*table\s*\d+",
            r"^\s*page\s*\d+\s*$",
        ]

    def segment_document(self, elements: List[Element]) -> List[Dict[str, Any]]:
        """
        Segments a document by title and then validates and refines the resulting sections.
        Returns sections containing the original elements.
        """
        logger.info(f"Attempting structural segmentation with `chunk_by_title` on {len(elements)} elements.")
        try:
            chunks = chunk_by_title(elements, max_characters=2048, combine_text_under_n_chars=128)
            
            initial_sections = []
            for chunk in chunks:
                if isinstance(chunk, CompositeElement):
                    section_elements = chunk.metadata.orig_elements
                else:
                    section_elements = [chunk]
                
                title = "Untitled Section"
                if section_elements:
                    try:
                        # Safely access the title attribute.
                        title = section_elements[0].metadata.title
                    except AttributeError:
                        # If the title attribute doesn't exist, keep the default.
                        pass

                initial_sections.append({"title": title, "elements": section_elements})

            refined_sections = self._validate_and_refine_sections(initial_sections)
            
            final_output = []
            for section in refined_sections:
                elements_as_dicts = []
                for el in section["elements"]:
                    element_dict = el.to_dict()
                    if 'metadata' in element_dict and 'coordinates' in element_dict['metadata']:
                        del element_dict['metadata']['coordinates']
                    elements_as_dicts.append(element_dict)
                
                final_output.append({
                    "title": section["title"],
                    "elements": elements_as_dicts
                })
            
            logger.info(f"Structural segmentation complete. Found {len(final_output)} refined sections.")
            return final_output

        except Exception as e:
            logger.error(f"An error occurred during structural segmentation: {e}", exc_info=True)
            return []

    def _validate_and_refine_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs a series of validation and refinement steps on sections containing elements.
        """
        logger.info(f"Starting validation and refinement for {len(sections)} initial sections.")

        # 1. Filter out irrelevant titles
        filtered_sections = [s for s in sections if not any(re.match(p, s['title'].strip(), re.IGNORECASE) for p in self.irrelevant_title_patterns)]
        
        # 2. Merge sections with the same title
        merged_sections_map = {}
        for section in filtered_sections:
            title_key = section['title'].strip().lower()
            if title_key in merged_sections_map:
                merged_sections_map[title_key]['elements'].extend(section['elements'])
            else:
                merged_sections_map[title_key] = section
        deduped_sections = list(merged_sections_map.values())

        # 3. Balance sections: merge very short sections into the previous one
        balanced_sections = []
        if deduped_sections:
            balanced_sections.append(deduped_sections[0])
            for i in range(1, len(deduped_sections)):
                section = deduped_sections[i]
                section_text = "\n\n".join([el.text for el in section['elements']])
                if len(section_text) < 150:
                    # Prepend the title of the short section to its elements before merging
                    title_element = Text(text=f"\n\n--- {section['title']} ---\n\n")
                    balanced_sections[-1]['elements'].append(title_element)
                    balanced_sections[-1]['elements'].extend(section['elements'])
                else:
                    balanced_sections.append(section)

        # 4. Coherence Check: Ensure content matches the title
        final_sections = []
        for section in balanced_sections:
            title = section['title']
            elements = section['elements']
            content = "\n\n".join([el.text for el in elements])
            
            if not title or not content.strip() or title.strip().lower() == 'untitled section':
                continue

            try:
                title_embedding = self.embedder.get_text_embedding(title)
                content_embedding = self.embedder.get_text_embedding(content)
                similarity = self.embedder.similarity(title_embedding, content_embedding)
                
                if similarity < 0.4:
                    logger.warning(f"Low coherence score ({similarity:.2f}) for section '{title}'. Discarding.")
                else:
                    final_sections.append(section)
            except Exception as e:
                logger.error(f"Could not perform coherence check for section '{title}': {e}")
                # Keep the section if the check fails, as it's better to have it than lose it
                final_sections.append(section)

        return final_sections