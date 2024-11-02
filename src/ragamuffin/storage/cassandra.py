import logging

import cassio
from cassandra.cluster import Cluster
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.readers.base import BaseReader
from llama_index.vector_stores.cassandra import CassandraVectorStore

logger = logging.getLogger(__name__)

# $ brew install cassandra
# $ brew services start cassandra
# $ cqlsh
# cqlsh> CREATE KEYSPACE zoterochat WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};


class CassandraStorage:
    def __init__(self, cluster_ip: str, keyspace: str):
        self.cluster_ip = cluster_ip
        self.keyspace = keyspace

        self.cluster = Cluster([self.cluster_ip])
        self.session = self.cluster.connect()
        cassio.init(session=self.session, keyspace=keyspace)

        self.vector_store = CassandraVectorStore(table="vector_store", embedding_dimension=1536)

    def generate_index(self, reader: BaseReader) -> BaseIndex:
        """Load the documents and create a RAG index."""
        documents = reader.load_data()
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist()
        return index

    def load_index(self) -> BaseIndex:
        """Load the index from storage."""
        return VectorStoreIndex.from_vector_store(self.vector_store)
