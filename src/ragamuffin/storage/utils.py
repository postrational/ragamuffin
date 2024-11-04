import logging

from ragamuffin.settings import get_settings
from ragamuffin.storage.cassandra import CassandraStorage
from ragamuffin.storage.file import FileStorage
from ragamuffin.storage.interface import Storage

logger = logging.getLogger(__name__)


def get_storage() -> Storage:
    """Get the storage implementation based on the environment."""
    settings = get_settings()
    storage_type = settings.get("storage_type")

    if storage_type == "file":
        return FileStorage()

    if storage_type == "cassandra":
        logger.info("Connecting to the Cassandra cluster...")

        return CassandraStorage(
            cluster_ip=settings.get("cassandra_cluster_ip"), keyspace=settings.get("cassandra_keyspace")
        )

    raise ValueError(f"Unknown storage type '{storage_type}'.")
