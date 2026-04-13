from abc import ABC, abstractmethod
from typing import List, Protocol
from unstructured.documents.elements import Element


class Loader(ABC):
    @abstractmethod
    def process(self) -> List[Element]:
        """Processes a file and returns a list of unstructured Elements."""
        pass


class LoaderType(Protocol):
    """Defines the constructor interface for loader classes."""
    def __call__(self, file_path: str) -> 'Loader':
        ...