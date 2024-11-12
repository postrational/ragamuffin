import logging
import shutil
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.readers.base import BaseReader

from ragamuffin.models.model_picker import configure_llamaindex_embedding_model
from ragamuffin.settings import get_settings
from ragamuffin.storage.interface import Storage

logger = logging.getLogger(__name__)


class FileStorage(Storage):
    def __init__(self):
        settings = get_settings()
        self.embedding_dimension = settings.get("embedding_dimension")

        settings = get_settings()
        self.persist_dir = Path(settings.get("data_dir")) / "storage"

    def get_agent_storage_dir(self, agent_name: str) -> Path:
        """Get the storage directory for the agent."""
        persist_dir = self.persist_dir / agent_name
        persist_dir.mkdir(parents=True, exist_ok=True)
        return persist_dir

    def generate_index(self, agent_name: str, reader: BaseReader) -> BaseIndex:
        """Load the documents and create a RAG index."""
        logger.info("Loading documents...")
        documents = reader.load_data()

        # Configure chunking settings
        configure_llamaindex_embedding_model()

        # Build the index from documents and persist to disk
        logger.info("Generating RAG embeddings...")
        index = VectorStoreIndex.from_documents(documents)
        logger.info("Storing the index in the file system...")
        index.storage_context.persist(persist_dir=self.get_agent_storage_dir(agent_name))
        return index

    def load_index(self, agent_name: str) -> BaseIndex:
        """Load the index from storage."""
        persist_dir = str(self.get_agent_storage_dir(agent_name))
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)

    def list_agents(self) -> list[str]:
        """Get the list of agents."""
        return [path.name for path in self.persist_dir.iterdir() if path.is_dir()]

    def delete_agent(self, agent_name: str) -> None:
        """Delete the agent from storage."""
        if agent_name not in self.list_agents():
            logger.warning(f"Agent '{agent_name}' does not exist.")
            return

        agent_dir = self.get_agent_storage_dir(agent_name)
        shutil.rmtree(agent_dir)
        logger.info(f"Deleted agent storage directory: {agent_dir}")
