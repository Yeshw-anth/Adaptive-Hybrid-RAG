
import logging
from typing import List, Dict, Any
from unstructured.documents.elements import Table, NarrativeText, Title, ListItem, Formula

from config import ENABLE_IMAGE_CAPTIONING
from multimodal.image_captioner import generate_caption

# Configure logging
logger = logging.getLogger(__name__)

def route_content(elements: List) -> List[Dict[str, Any]]:
    """
    Routes partitioned elements to different processing strategies based on their type.

    Args:
        elements (List): A list of elements from unstructured.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a content element
                               with its type and raw content.
    """
    routed_elements = []
    for element in elements:
        element_type = "text"  # Default type
        content = element.text
        metadata = element.metadata

        if isinstance(element, Table):
            element_type = "table"
            # Pass the original element for the table processor
            content = element
            logging.info("Routed a 'table' content type.")
        elif isinstance(element, (NarrativeText, Title, ListItem)):
            element_type = "text"
        elif isinstance(element, Formula):
            element_type = "formula"
            content = element.text  # Preserve formula as text
            logging.info("Routed a 'formula' content type.")
        elif "image" in str(type(element)).lower():
            element_type = "image"
            if ENABLE_IMAGE_CAPTIONING:
                image_path = metadata.get("filename") # unstructured provides filename for images
                if image_path:
                    logging.info(f"Image detected. Generating caption for: {image_path}")
                    caption = generate_caption(image_path)
                    if caption:
                        content = caption
                    else:
                        logging.warning(f"Could not generate caption for {image_path}. Skipping.")
                        continue # Skip this element if captioning fails
                else:
                    logging.warning("Image element found but no path available. Skipping.")
                    continue
            else:
                logging.info("Image detected, but image captioning is disabled. Skipping.")
                continue # Skip image elements if feature is disabled

        routed_elements.append({
            "type": element_type,
            "content": content,
            "metadata": metadata
        })
    
    logging.info(f"Content routing complete. Processed {len(routed_elements)} elements out of {len(elements)} total.")
    return routed_elements