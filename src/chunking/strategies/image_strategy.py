import logging
from typing import List, Dict, Any
from llama_index.core.schema import TextNode

from src.chunking.strategies.base_strategy import BaseChunkingStrategy
from src.data.schemas import DocumentMetadata
from src.experimental.multimodal.image_captioner import extract_structure_from_image
from src.core.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class ImageChunkingStrategy(BaseChunkingStrategy):
    """
    A strategy for processing image content. It generates a descriptive caption
    for an image and creates a single, self-contained TextNode from it.
    """
    def __init__(self):
        self.llm_client = OllamaClient()

    async def process(self, elements: List[Any], file_path: str) -> List[TextNode]:
        """
        Processes a section containing image elements.
        
        For each image, it generates a caption and creates a single TextNode.
        """
        nodes = []
        # The loader now provides the image path directly in the element's metadata
        image_paths = [el.metadata.get("image_path") for el in elements if el.metadata.get("image_path")]

        if not image_paths:
            logger.warning("Image strategy called, but no image paths found in section elements.")
            return []

        for image_path in image_paths:
            # Generate a real caption using the VLM
            structured_content = await extract_structure_from_image(image_path=image_path, llm_client=self.llm_client)
            
            if not structured_content:
                logger.warning(f"Could not generate caption for image {image_path}. Skipping.")
                continue

            # Combine the extracted content to form a single "caption"
            caption = " ".join([element['content'] for element in structured_content if 'content' in element and isinstance(element['content'], str)])

            logger.info(f"Creating node for image '{image_path}' with real caption.")
            
            # The text of the node IS the caption, which is what gets embedded
            node = TextNode(
                text=caption,
                metadata={
                    "is_parent": True,
                    "content_type": "image",
                    "image_path": image_path,
                    "image_caption": caption
                }
            )
            nodes.append(node)
            
        return nodes