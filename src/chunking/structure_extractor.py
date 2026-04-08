import logging
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
import os


logger = logging.getLogger(__name__)

class StructureExtractor:
    """
    Performs robust, dynamic sectioning on a document.
    It identifies headings to create logical, combined sections for downstream chunking.
    If no headings are found, it falls back to treating the document as a single section.
    """
    def _get_font_info(self, element: Any) -> Dict[str, Any]:
        """
        Extracts font information from element metadata using reliable, structured fields.
        Infers font size from the element's coordinates and bolding from `emphasized_text_contents`.
        """
        font_info = {"size": None, "bold": False}

        # 1. Infer font size from the height of the element's bounding box
        if hasattr(element, 'metadata') and hasattr(element.metadata, 'coordinates') and element.metadata.coordinates:
            points = element.metadata.coordinates.points
            if points and len(points) > 1:
                # Assumes points are ((x1, y1), (x2, y2), ...)
                # Height is the difference in the y-coordinates
                y_coords = [p[1] for p in points]
                height = max(y_coords) - min(y_coords)
                font_info["size"] = height
        
        # 2. Determine if the text is bold by checking the 'emphasized_text_contents'
        if hasattr(element, 'metadata') and hasattr(element.metadata, 'emphasized_text_contents'):
            emphasized_text = element.metadata.emphasized_text_contents
            if emphasized_text and element.text.strip() in emphasized_text:
                font_info["bold"] = True
                
        return font_info

    def _is_heading(self, element: Any, avg_font_size: float = None) -> bool:
        """
        Determines if an element is a heading using a scoring system based on
        category, style (font size, bold), and text patterns.
        """
        text = element.text.strip()
        if not text:
            return False

        score = 0
        
        # 1. Category-based scoring (highest weight)
        if hasattr(element, 'category') and element.category == "Title":
            score += 50
            logger.debug(f"Heading score +50 (Category='Title') for: '{text}'")

        # 2. Style-based scoring
        font_info = self._get_font_info(element)
        if font_info["bold"]:
            score += 20
            logger.debug(f"Heading score +20 (Bold) for: '{text}'")
        
        if font_info["size"] and avg_font_size:
            if font_info["size"] > avg_font_size * 1.15: # At least 15% larger
                score += 25
                logger.debug(f"Heading score +25 (Font size > avg) for: '{text}'")

        # 3. Content-based scoring
        if text.isupper() and len(text.split()) < 5:
            score += 10 # Lower score to avoid false positives on acronyms
            logger.debug(f"Heading score +10 (ALL CAPS) for: '{text}'")
            
        if text.istitle() and len(text.split()) < 5:
            score += 10
            logger.debug(f"Heading score +10 (Title Case) for: '{text}'")
            
        # Regex for patterns like "Chapter 1", "Section A", etc.
        import re
        if re.match(r'^(Chapter|Section|Part)\s+([A-Za-z0-9]+)', text):
            score += 40
            logger.debug(f"Heading score +40 (Regex pattern) for: '{text}'")

        # Final decision based on threshold
        is_heading = score >= 40
        if is_heading:
            logger.info(f"Element classified as HEADING (Score: {score}): '{text}'")
        return is_heading

    def extract_elements(self, file_path: str) -> List[Any]:
        """
        Partitions a document and returns the raw list of unstructured elements.
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            logger.info(f"Partitioning file with  auto strategy.")
            
            if file_extension == ".pdf":
                elements = partition_pdf(file_path, strategy="auto", infer_table_structure=True)
            else:
                elements = partition(file_path, strategy="auto")

            logger.info(f"Successfully partitioned file into {len(elements)} raw elements.")
            return elements
        except Exception as e:
            logger.error(f"Failed to partition file {file_path} for raw element extraction: {e}", exc_info=True)
            return []

    def extract_sections(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Partitions a document and groups its elements into logical sections based on headings.
        If no headings are found, it returns the whole document as one section.
        """
        try:
            logger.info(f"Starting structure extraction for file: {file_path}")
            elements = self.extract_elements(file_path)
            if not elements:
                return []
        except Exception as e:
            logger.error(f"Failed during element extraction for {file_path}: {e}", exc_info=True)
            return []

        # --- First Pass: Calculate average font size for style-based heading detection ---
        total_font_size = 0
        font_size_count = 0
        for element in elements:
            font_info = self._get_font_info(element)
            if font_info["size"]:
                total_font_size += font_info["size"]
                font_size_count += 1
        
        avg_font_size = (total_font_size / font_size_count) if font_size_count > 0 else None
        if avg_font_size:
            logger.info(f"Calculated average font size: {avg_font_size:.2f}px")

        # --- Second Pass: Identify sections ---
        sections = []
        current_title = "Introduction"  # Default for content before the first heading
        current_elements = []

        def finalize_section():
            """Helper to finalize the current section and add it to the list."""
            if not current_elements:
                return
            
            cleaned_elements = [el for el in current_elements if el.text.strip()]
            if cleaned_elements:
                sections.append({"title": current_title, "elements": cleaned_elements})
                logger.debug(f"Finalized section: '{current_title}' with {len(cleaned_elements)} elements.")

        for element in elements:
            text = element.text.strip()
            if not text:
                continue

            if self._is_heading(element, avg_font_size=avg_font_size):
                finalize_section()  # Finalize the previous section
                current_title = text  # Start a new one
                current_elements = [element]  # Add heading to the new section
            else:
                current_elements.append(element)
        
        finalize_section()  # Ensure the last section is added

        # --- Fallback Mechanism ---
        if not sections and elements:
            logger.warning(f"No headings found in {os.path.basename(file_path)}. "
                           f"Falling back to a single section for the whole document.")
            all_cleaned_elements = [el for el in elements if el.text.strip()]
            if all_cleaned_elements:
                sections.append({
                    "title": "Full Document",
                    "elements": all_cleaned_elements
                })

        logger.info(f"Extraction complete: Found {len(sections)} logical sections.")
        return sections