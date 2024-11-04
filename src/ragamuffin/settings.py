import os

from platformdirs import user_data_dir


def get_settings() -> dict[str, str]:
    """Get settings from environment variables."""
    return {
        "storage_type": os.environ.get("RAGAMUFFIN_STORAGE_TYPE", "file"),
        "data_dir": os.environ.get("RAGAMUFFIN_DATA_DIR", user_data_dir("ragamuffin")),
        "cassandra_cluster_ip": os.environ.get("CASSANDRA_CLUSTER", "127.0.0.1"),
        "cassandra_keyspace": os.environ.get("CASSANDRA_KEYSPACE", "ragamuffin"),
        "zotero_library_id": os.environ.get("ZOTERO_LIBRARY_ID"),
        "zotero_api_key": os.environ.get("ZOTERO_API_KEY"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
    }
