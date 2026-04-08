import os
import uuid
import logging
from typing import List, Dict, Any
import fitz  # PyMuPDF
from PIL import Image

from .base_loader import BaseLoader
from src.core.llm.ollama_client import OllamaClient
from src.experimental.multimodal.image_captioner import extract_structure_from_image
from src.config import settings

logger = logging.getLogger(__name__)

class PdfLoader(BaseLoader):
    """
    A loader for PDF files that uses a hybrid approach. It first tries to
    extract digital text and, if a page appears to be scanned (image-based),
    it falls back to a multimodal LLM for OCR and structure extraction.
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        self.image_output_dir = settings.IMAGE_OUTPUT_DIR
        os.makedirs(self.image_output_dir, exist_ok=True)
        logger.info(f"PDF Loader initialized. Image output directory: {self.image_output_dir}")

    async def load(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Loads a PDF file, processing each page with a hybrid text/image strategy.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        logger.debug(f"Loading PDF file: {file_path}")
        all_elements = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_elements = await self._process_page(page, file_path, page_num)
                all_elements.extend(page_elements)
            
            doc.close()
            logger.info(f"Successfully processed {len(all_elements)} elements from {file_path}.")
            return all_elements

        except Exception as e:
            logger.error(f"Failed to load or process PDF {file_path}: {e}", exc_info=True)
            return []

    async def _process_page(self, page: fitz.Page, file_path: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Processes a single page, deciding whether to use text extraction or image-based structure analysis.
        """
        # Heuristic: If a page has very little text, it's likely a scanned image.
        # The threshold can be tuned. 100 characters is a reasonable starting point.
        text_blocks = page.get_text("blocks")
        if text_blocks:
            logger.debug(f"Page {page_num + 1}: Detected {len(text_blocks)} text blocks. Using block-based extraction.")
            return [self._create_text_element(block[4], file_path, page_num, i) for i, block in enumerate(text_blocks)]
        else:
            logger.warning(f"Page {page_num + 1}: Low text content detected. Attempting image-based extraction.")
            image_path = self._render_page_as_image(page, file_path, page_num)
            if image_path:
                # Use the multimodal LLM to extract structured elements from the image
                image_elements = await extract_structure_from_image(image_path, self.llm_client)
                # Add page-specific metadata to the elements returned by the LLM
                for i, element in enumerate(image_elements):
                    element['metadata'] = {
                        "file_path": file_path,
                        "page_number": page_num + 1,
                        "element_type_from_llm": element.get('type', 'unknown'),
                        "chunk_id": f"{os.path.basename(file_path)}-p{page_num+1}-{i}"
                    }
                return image_elements
            else:
                logger.error(f"Could not render page {page_num + 1} as an image.")
                return []

    def _render_page_as_image(self, page: fitz.Page, file_path: str, page_num: int) -> str:
        """Renders a PyMuPDF page object to a high-resolution PNG image."""
        try:
            # Use a high DPI for better OCR quality
            pix = page.get_pixmap(dpi=300)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            base_filename = os.path.basename(file_path)
            image_filename = f"{base_filename}_page_{page_num + 1}.png"
            image_path = os.path.join(self.image_output_dir, image_filename)
            
            image.save(image_path)
            logger.info(f"Saved page {page_num + 1} as image to {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Error rendering page {page_num + 1} to image: {e}")
            return ""

    def _create_text_element(self, text: str, file_path: str, page_num: int, block_idx: int) -> Dict[str, Any]:
        """Creates a standard document element for a block of text."""
        chunk_id = f"{os.path.basename(file_path)}-p{page_num+1}-b{block_idx}"
        return {
            "text": text,
            "metadata": {
                "file_path": file_path,
                "page_number": page_num + 1,
                "content_type": "text",
                "chunk_id": chunk_id
            }
        }