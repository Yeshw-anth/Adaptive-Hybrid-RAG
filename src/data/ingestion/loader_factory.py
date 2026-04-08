import os
from typing import Type
from .base_loader import BaseLoader
from .text_loader import TextLoader
from .pdf_loader import PdfLoader
from .csv_loader import CsvLoader
from .docx_loader import DocxLoader
from .html_loader import HtmlLoader
from .json_loader import JsonLoader
from .md_loader import MdLoader
from .pptx_loader import PptxLoader
from src.core.llm.ollama_client import OllamaClient

# A mapping from file extensions to their corresponding loader classes
LOADER_MAPPING = {
    ".csv": CsvLoader,
    ".docx": DocxLoader,
    ".html": HtmlLoader,
    ".json": JsonLoader,
    ".md": MdLoader,
    ".pdf": PdfLoader,
    ".pptx": PptxLoader,
    ".txt": TextLoader,
}

# Loaders that require the llm_client instance
CLIENT_AWARE_LOADERS = {".pdf"}

def get_loader(file_path: str, llm_client: OllamaClient = None) -> BaseLoader:
    """
    Selects the appropriate document loader based on the file extension.
    
    Args:
        file_path: The path to the file to be loaded.
        llm_client: An optional OllamaClient instance, required by some loaders.

    Returns:
        An instance of a BaseLoader subclass.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    LoaderClass = LOADER_MAPPING.get(ext)
    
    if not LoaderClass:
        # Default to TextLoader for any unhandled extension
        return TextLoader()

    if ext in CLIENT_AWARE_LOADERS:
        if not llm_client:
            raise ValueError(f"The loader for '{ext}' files requires an LLM client.")
        return LoaderClass(llm_client=llm_client)
    
    return LoaderClass()