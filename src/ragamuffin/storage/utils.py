import logging

from ragamuffin.error_handling import ConfigurationError, ensure_string
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
        ip = ensure_string(settings.get("cassandra_cluster_ip"))
        keyspace = ensure_string(settings.get("cassandra_keyspace"))
        return CassandraStorage(cluster_ip=ip, keyspace=keyspace)

    raise ConfigurationError(f"Unknown storage type '{storage_type}'.")
