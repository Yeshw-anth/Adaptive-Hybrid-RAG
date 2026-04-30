from src.core.logging_config import logger
from typing import List, Dict, Any
from llama_index.core.schema import TextNode
from unstructured.documents.elements import Element, Table

from src.chunking.segmentation_evaluator import SegmentationEvaluator
from src.chunking.strategy_factory import StrategyFactory


class ChunkingEngine:
    """
    Orchestrates the document chunking process. It first uses a segmentation evaluator
    to divide the document into high-level semantic or structural sections. Then, it
    dispatches the elements within each section to content-specific chunking strategies
    (e.g., for text, tables) to produce fine-grained TextNode objects.
    """
    def __init__(self, segmentation_evaluator: SegmentationEvaluator, strategy_factory: StrategyFactory):
        """
        Initializes the ChunkingEngine.

        Args:
            segmentation_evaluator: The evaluator that runs segmentation strategies.
            strategy_factory: A factory to get the appropriate chunking strategy for each element type.
        """
        self.segmentation_evaluator = segmentation_evaluator
        self.strategy_factory = strategy_factory
        logger.info("ChunkingEngine initialized with segmentation evaluator and strategy factory.")

    async def chunk_document(
        self,
        elements: List[Element],
        file_path: str,
        initial_strategy: str = 'structural'
    ) -> List[TextNode]:
        """
        Chunks a document by first segmenting it into sections and then processing
        the elements within each section using content-specific strategies.

        Args:
            elements: A list of unstructured document elements.
            file_path: The path to the original document.
            initial_strategy: The preferred strategy for the evaluator to start with.

        Returns:
            A list of TextNode objects representing the chunked document.
        """
        logger.info(f"Starting chunking for '{file_path}' with initial strategy: '{initial_strategy}'")
        
        sections, successful_strategy = self.segmentation_evaluator.segment_document(
            elements, file_path, initial_strategy=initial_strategy
        )

        if not sections:
            logger.error(f"Segmentation evaluator returned no sections for {file_path} (strategy: {successful_strategy}).")
            return []

        all_nodes = []
        for section in sections:
            section_title = section.get('title', 'Untitled Section')
            section_elements = section.get('elements', [])
            
            if not section_elements:
                continue

            # Group consecutive elements of the same type to be processed together
            element_groups = self._group_elements_by_strategy(section_elements)

            for strategy_name, group in element_groups:
                strategy = self.strategy_factory.get_strategy(strategy_name)
                
                # The process method of a strategy should handle one or more elements
                nodes = await strategy.process(group, file_path)
                
                # Add section-level metadata to each node
                for node in nodes:
                    node.metadata.update({
                        "file_path": file_path,
                        "section_title": section_title,
                        "segmentation_strategy": successful_strategy,
                    })
                all_nodes.extend(nodes)

        logger.info(f"Successfully chunked '{file_path}' into {len(all_nodes)} nodes using '{successful_strategy}' strategy.")
        return all_nodes

    def _group_elements_by_strategy(self, elements: List[Element]) -> List[tuple[str, List[Element]]]:
        """
        Groups consecutive elements that should be processed by the same strategy.
        
        For example, multiple `NarrativeText` elements would be grouped together for the 'text' strategy.
        A `Table` element would be in its own group for the 'table' strategy.
        """
        if not elements:
            return []

        groups = []
        current_strategy = None
        current_group = []

        for el in elements:
            strategy_name = self._determine_strategy_for_element(el)
            
            if strategy_name != current_strategy and current_group:
                groups.append((current_strategy, current_group))
                current_group = []
            
            current_strategy = strategy_name
            current_group.append(el)
        
        if current_group:
            groups.append((current_strategy, current_group))
            
        return groups

    def _determine_strategy_for_element(self, element: Element) -> str:
        """Determines which strategy to use for a given element."""
        if isinstance(element, Table):
            return 'table'
        # Add more conditions for images, code, etc.
        # Default to text strategy
        return 'text'