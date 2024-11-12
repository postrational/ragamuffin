import os

from platformdirs import user_data_dir


def get_settings() -> dict[str, str | int | bool | None]:
    """Get settings from environment variables."""
    settings = {
        "storage_type": os.environ.get("RAGAMUFFIN_STORAGE_TYPE", "file"),
        "data_dir": os.environ.get("RAGAMUFFIN_DATA_DIR", user_data_dir("ragamuffin")),
        "llm_model": os.environ.get("RAGAMUFFIN_LLM_MODEL", "openai/gpt-4o-mini"),
        # Local model: "huggingface.co/BAAI/bge-m3", uses 1024-dimensional embeddings
        "embedding_model": os.environ.get("RAGAMUFFIN_EMBEDDING_MODEL", "openai/text-embedding-ada-002"),
        "embedding_dimension": os.environ.get("RAGAMUFFIN_EMBEDDING_DIMENSION", 1536),
        "debug_mode": os.environ.get("RAGAMUFFIN_DEBUG", False),
        "cassandra_cluster_ip": os.environ.get("CASSANDRA_CLUSTER", "127.0.0.1"),
        "cassandra_keyspace": os.environ.get("CASSANDRA_KEYSPACE", "ragamuffin"),
        "zotero_library_id": os.environ.get("ZOTERO_LIBRARY_ID"),
        "zotero_api_key": os.environ.get("ZOTERO_API_KEY"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
    }

    # Handle boolean values
    for key in ["debug_mode"]:
        value = settings[key]
        if isinstance(value, str):
            settings[key] = value.lower() in ["true", "1", "yes"]

    # Handle integer values
    for key in ["embedding_dimension"]:
        value = settings[key]
        if isinstance(value, str):
            settings[key] = int(value)

    return settings
