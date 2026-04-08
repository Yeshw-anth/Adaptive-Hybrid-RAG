from typing import List, Dict, Any
from unstructured.documents.elements import Element

def convert_to_dialogue_format(elements: List[Element], metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Converts a list of unstructured Element objects into our standard
    dialogue-style dictionary format.

    Args:
        elements: A list of unstructured Element objects.
        metadata: Optional base metadata to include in each element.

    Returns:
        A list of dictionaries, each representing an element.
    """
    if metadata is None:
        metadata = {}
        
    dialogue = []
    for i, element in enumerate(elements):
        element_dict = element.to_dict()
        
        # Create a unique ID for the chunk
        file_path = element_dict.get("metadata", {}).get("filename", "unknown_file")
        page_number = element_dict.get("metadata", {}).get("page_number", 1)
        chunk_id = f"{file_path}-p{page_number}-{i}"

        dialogue.append({
            "type": element_dict.get("type", "Unknown"),
            "content": element_dict.get("text", ""),
            "metadata": {
                **metadata,
                "source_metadata": element_dict.get("metadata", {}),
                "element_id": element_dict.get("id", None),
                "chunk_id": chunk_id
            }
        })
    return dialogue