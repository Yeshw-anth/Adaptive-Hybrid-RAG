import logging
import base64
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from src.core.llm.ollama_client import OllamaClient
from src.config import settings

logger = logging.getLogger(__name__)

# --- Pydantic Models for Strict Output Validation ---

class Element(BaseModel):
    type: str = Field(..., description="The type of the document element.")
    content: Any = Field(..., description="The content of the element.")

class StructuredImageResponse(BaseModel):
    elements: List[Element] = Field(..., description="A list of structured elements extracted from the image.")

# --- Main Extractor Function ---

async def extract_structure_from_image(
    image_path: str, 
    llm_client: OllamaClient
) -> List[Dict[str, Any]]:
    """
    Uses a multimodal LLM to extract structured content from an image of a document page.

    Args:
        image_path: The local path to the image file.
        llm_client: An instance of the OllamaClient.

    Returns:
        A list of dictionaries, where each dictionary represents a structured element.
        Returns an empty list if processing fails.
    """
    logger.info(f"Extracting structure from image: {image_path}")
    
    system_prompt = """
You are an expert document digitization and structuring system. Your task is to analyze the provided document image and convert it into a structured JSON format. You must identify all logical elements on the page, such as titles, paragraphs (narrative text), lists, and tables.

RULES:
1. Your response MUST be a single, valid JSON object.
2. The JSON object must have a single key: "elements".
3. "elements" must be a list of objects.
4. Each object in the list must have a "type" and a "content" key.
5. Valid "type" values are: "Title", "NarrativeText", "ListItem", "Table".
6. For "Table" elements, the "content" should be a 2D array of strings representing the cells (a list of lists).
7. For all other elements, "content" should be the text of the element.
8. Be precise in your extraction. Do not hallucinate content. If a table is present, extract it fully.

Analyze the user-provided image and provide ONLY the JSON output.
"""

    try:
        # 1. Encode image to base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 2. Call the multimodal LLM
        response_text = await llm_client.generate_with_image(
            model=settings.MULTIMODAL_LLM_MODEL,
            prompt="Analyze this document image and extract its structure.",
            system_prompt=system_prompt,
            images_base64=[image_base64]
        )

        # 3. Parse and validate the response
        json_response = llm_client.clean_json_response(response_text)
        validated_response = StructuredImageResponse.model_validate_json(json_response)
        
        logger.info(f"Successfully extracted {len(validated_response.elements)} elements from {image_path}.")
        return [el.model_dump() for el in validated_response.elements]

    except FileNotFoundError:
        logger.error(f"Image file not found at path: {image_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM response for image {image_path}.")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during image structure extraction for {image_path}: {e}")
        return []