from pathlib import Path

from ragamuffin.libraries.files import LocalLibrary
from ragamuffin.storage.file import FileStorage
from tests.utils import env_vars, seed


@seed(42)
def test_file_storage_create_agent():
    test_data_path = Path(__file__).parent / "data" / "udhr"
    library = LocalLibrary(str(test_data_path))
    reader = library.get_reader()

    storage = FileStorage()
    agent_name = "test_agent"
    with env_vars(
        RAGAMUFFIN_EMBEDDING_DIMENSION="312",
        RAGAMUFFIN_EMBEDDING_MODEL="huggingface.co/huawei-noah/TinyBERT_General_4L_312D",
    ):
        storage.generate_index(agent_name, reader=reader)

    assert agent_name in storage.list_agents()

    index = storage.load_index(agent_name)
    ingested_doc_metadata = list(index.ref_doc_info.values())[0].metadata
    assert ingested_doc_metadata["file_name"] == "udhr-en.pdf"
    assert ingested_doc_metadata["file_type"] == "application/pdf"
    assert ingested_doc_metadata["page_label"] == "1"

    storage.delete_agent(agent_name)
    assert agent_name not in storage.list_agents()
