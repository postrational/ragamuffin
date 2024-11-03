import logging

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from typing_extensions import override

from ragamuffin.libraries.interface import Library

logger = logging.getLogger(__name__)


class LocalLibrary(Library):
    def __init__(self, library_dir: str, **kwargs: dict):
        super().__init__(**kwargs)
        self.library_dir = library_dir

    @override
    def get_reader(self) -> BaseReader:
        """Get a Llama Index reader for the library files."""
        return SimpleDirectoryReader(input_dir=self.library_dir, recursive=True)
