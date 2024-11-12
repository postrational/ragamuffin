import logging
import re
import sys

import cassio
from cassandra.cluster import Cluster
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.readers.base import BaseReader
from llama_index.vector_stores.cassandra import CassandraVectorStore

from ragamuffin.error_handling import ensure_int
from ragamuffin.models.model_picker import configure_llamaindex_embedding_model
from ragamuffin.settings import get_settings
from ragamuffin.storage.interface import Storage

logger = logging.getLogger(__name__)


class CassandraStorage(Storage):
    def __init__(self, cluster_ip: str, keyspace: str):
        self.cluster_ip = cluster_ip
        self.keyspace = keyspace
        self.cluster = Cluster([self.cluster_ip])
        self.session = self.cluster.connect()
        cassio.init(session=self.session, keyspace=keyspace)

    def _validate_agent_name(self, agent_name: str) -> None:
        if not bool(re.match(r"^[a-z_][a-z0-9_]{0,47}$", agent_name)):
            logger.error(f"Invalid agent name: {agent_name}")
            logger.info(
                "Agent names must start with a letter or underscore, contain alphanumeric "
                "lowercase characters and underscores, and be 1-48 characters long."
            )
            sys.exit(4)

    def generate_index(self, agent_name: str, reader: BaseReader) -> BaseIndex:
        """Load the documents and create a RAG index."""
        self._validate_agent_name(agent_name)
        logger.info("Loading documents...")
        documents = reader.load_data()

        logger.info("Generating RAG embeddings...")
        settings = get_settings()
        embed_dim = ensure_int(settings.get("embedding_dimension"))
        vector_store = CassandraVectorStore(table=agent_name, embedding_dimension=embed_dim)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        configure_llamaindex_embedding_model()

        logger.info("Storing the index in Cassandra...")
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist()
        return index

    def load_index(self, agent_name: str) -> BaseIndex:
        """Load the index from storage."""
        settings = get_settings()
        embed_dim = ensure_int(settings.get("embedding_dimension"))
        vector_store = CassandraVectorStore(table=agent_name, embedding_dimension=embed_dim)
        return VectorStoreIndex.from_vector_store(vector_store)

    def list_agents(self) -> list[str]:
        """Get the list of agents."""
        query = "SELECT table_name FROM system_schema.tables WHERE keyspace_name = %s"
        rows = self.session.execute(query, [self.keyspace])
        return [row.table_name for row in rows]

    def delete_agent(self, agent_name: str) -> None:
        """Delete the agent table from Cassandra keyspace."""
        self._validate_agent_name(agent_name)

        query = "SELECT table_name FROM system_schema.tables WHERE keyspace_name = %s AND table_name = %s"
        result = self.session.execute(query, (self.keyspace, agent_name))
        if not result.one():
            logger.warning(f"Agent '{agent_name}' does not exist.")
            return

        query = f"DROP TABLE {self.keyspace}.{agent_name}"
        self.session.execute(query)
        logger.info(f"Deleted agent '{agent_name}'.")
