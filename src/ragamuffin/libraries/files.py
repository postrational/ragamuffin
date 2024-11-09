import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from typing_extensions import override

from ragamuffin.libraries.interface import Library

logger = logging.getLogger(__name__)


class LocalLibrary(Library):
    def __init__(self, library_dir: str):
        self.library_source = library_dir

    @override
    def get_reader(self) -> BaseReader:
        """Get a Llama Index reader for the library files."""
        # Check if library source is a single file
        if Path(self.library_source).is_file():
            return SimpleDirectoryReader(input_files=[self.library_source])

        return SimpleDirectoryReader(input_dir=self.library_source, recursive=True)
