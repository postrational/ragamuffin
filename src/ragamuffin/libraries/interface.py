import logging
from abc import ABC, abstractmethod

from llama_index.core.readers.base import BaseReader

logger = logging.getLogger(__name__)


class Library(ABC):
    def __init__(self, **kwargs: dict):
        raise NotImplementedError

    @abstractmethod
    def get_reader(self) -> BaseReader:
        """Get a Llama Index reader for the library files."""
        raise NotImplementedError
