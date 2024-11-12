from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

from ragamuffin.libraries.files import LocalLibrary
from tests.utils import seed


@seed(42)
def test_local_directory():
    test_data_path = Path(__file__).parent / "data" / "udhr"
    library = LocalLibrary(str(test_data_path))
    reader = library.get_reader()

    assert isinstance(reader, SimpleDirectoryReader)
    assert len(reader.list_resources()) == 2

    data = reader.load_data()
    assert len(data) == 9

    document = data[0]
    assert isinstance(document, Document)
    assert document.metadata["file_name"] == "udhr-en.pdf"
    assert document.metadata["file_type"] == "application/pdf"
    assert "progress and better standards of life in larger freedom" in document.text


def test_local_file():
    test_data_path = Path(__file__).parent / "data" / "udhr" / "udhr-en.pdf"
    library = LocalLibrary(str(test_data_path))
    reader = library.get_reader()

    assert isinstance(reader, SimpleDirectoryReader)
    assert len(reader.list_resources()) == 1

    data = reader.load_data()
    assert len(data) == 8
