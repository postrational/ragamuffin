import logging
import re

import cassio
from cassandra.cluster import Cluster
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.readers.base import BaseReader
from llama_index.vector_stores.cassandra import CassandraVectorStore

logger = logging.getLogger(__name__)


class CassandraStorage:
    def __init__(self, cluster_ip: str, keyspace: str, table: str):
        self.cluster_ip = cluster_ip
        self.keyspace = keyspace

        if not bool(re.fullmatch(r"\w+", table)):
            raise ValueError(f"Agent name must contain only alphanumeric characters and underscores, got: {table}")

        self.cluster = Cluster([self.cluster_ip])
        self.session = self.cluster.connect()
        cassio.init(session=self.session, keyspace=keyspace)

        self.vector_store = CassandraVectorStore(table=table, embedding_dimension=1536)

    def generate_index(self, reader: BaseReader) -> BaseIndex:
        """Load the documents and create a RAG index."""
        documents = reader.load_data()
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist()
        return index

    def load_index(self) -> BaseIndex:
        """Load the index from storage."""
        return VectorStoreIndex.from_vector_store(self.vector_store)
