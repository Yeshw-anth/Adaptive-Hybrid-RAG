import os
import logging
from typing import List
from itertools import groupby
from llama_index.core.schema import TextNode, Document
from llama_index.core.embeddings import BaseEmbedding

from src.chunking.strategy_factory import StrategyFactory
from src.chunking.structure_extractor import StructureExtractor
from src.chunking.metadata_enricher import MetadataEnricher

logger = logging.getLogger(__name__)

class ChunkingEngine:
    """
    The central orchestrator for the adaptive chunking process.
    This engine uses a multi-layered approach:
    1. It uses a SectionExtractor to identify logical sections in the document.
    2. If sectioning fails, it falls back to a simple text splitter.
    3. Within each section, it groups elements by content type (text, image, etc.).
    4. It delegates the chunking of each group to a specialized strategy.
    """

    def __init__(self, embedding_model: BaseEmbedding):
        self.strategy_factory = StrategyFactory(embedding_model=embedding_model)
        self.structure_extractor = StructureExtractor()
        self.metadata_enricher = MetadataEnricher()
        logger.info("ChunkingEngine initialized with StructureExtractor and MetadataEnricher.")

    async def chunk_document(self, file_path: str) -> List[TextNode]:
        """
        Processes a document from a file path using a section-aware, multi-modal approach.
        """
        logger.info(f"Starting chunking process for file: {file_path}")

        # 1. Identify logical sections using the StructureExtractor
        sections = self.structure_extractor.extract_sections(file_path)
        if not sections:
            logger.warning(f"No sections were extracted from {file_path}. Aborting chunking.")
            return []
        
        all_nodes = []

        # 2. Process each section (even if there's only one)
        for section in sections:
            section_title = section.get("title", "Untitled")
            section_elements = section.get("elements", [])
            logger.debug(f"Processing section: '{section_title}' with {len(section_elements)} elements.")
            
            # 3. Group elements by type WITHIN the section
            def get_element_type(element: Document):
                # The 'unstructured' library uses the 'category' attribute for element type.
                # We use getattr for safe access with a default value.
                return getattr(element.metadata, 'category', 'text')

            element_groups = groupby(section_elements, key=get_element_type)

            for element_type, group in element_groups:
                group_elements = list(group)
                logger.debug(f"  - Processing group of {len(group_elements)} elements of type '{element_type}'.")
                
                strategy = self.strategy_factory.get_strategy(element_type)
                
                # 4. Delegate to the appropriate strategy to get basic nodes
                group_nodes = await strategy.process(group_elements, file_path)
                
                # 5. Enrich nodes with centralized metadata
                enriched_nodes = self.metadata_enricher.enrich_nodes(
                    group_nodes,
                    file_name=os.path.basename(file_path),
                    file_path=file_path,
                    section_title=section_title
                )
                
                all_nodes.extend(enriched_nodes)
                logger.debug(f"  - Generated and enriched {len(enriched_nodes)} nodes for type '{element_type}'.")

        logger.info(f"Chunking complete for {file_path}. Generated {len(all_nodes)} total nodes.")
        return all_nodes