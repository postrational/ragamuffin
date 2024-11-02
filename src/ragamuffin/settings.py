import os


def get_settings() -> dict[str, str]:
    """Get settings from environment variables."""
    return {
        "cassandra_cluster_ip": os.environ.get("CASSANDRA_CLUSTER", "127.0.0.1"),
        "cassandra_keyspace": os.environ.get("CASSANDRA_KEYSPACE", "zoterochat"),
        "zotero_library_id": os.environ.get("ZOTERO_LIBRARY_ID"),
        "zotero_api_key": os.environ.get("ZOTERO_API_KEY"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
    }
