import logging

import click

from ragamuffin.libraries.zotero import ZoteroLibrary
from ragamuffin.settings import get_settings
from ragamuffin.storage.cassandra import CassandraStorage
from ragamuffin.webui.gradio_chat import GradioAgentChatUI

# @TODO: Add the ability to set similarity_top_k of the bot retriever
# llama_index.core.constants.DEFAULT_SIMILARITY_TOP_K = 6

logger = logging.getLogger(__name__)


@click.command()
@click.option("--regenerate", is_flag=True, help="Regenerate the index from scratch.")
def main(regenerate: bool) -> None:
    """Start the chat interface."""
    settings = get_settings()

    logger.info("Connecting to the Cassandra cluster...")
    storage = CassandraStorage(
        cluster_ip=settings.get("cassandra_cluster_ip"),
        keyspace=settings.get("cassandra_keyspace"),
    )

    if regenerate:
        library = ZoteroLibrary(
            library_id=settings["zotero_library_id"],
            api_key=settings["zotero_api_key"],
        )
        reader = library.get_reader()

        logger.info("Generating RAG embeddings...")
        index = storage.generate_index(reader)
    else:
        logger.info("Loading the RAG embedding index...")
        index = storage.load_index()

    logger.info("Starting the chat interface...")
    agent = index.as_chat_engine()
    webapp = GradioAgentChatUI(agent)
    webapp.run()


if __name__ == "__main__":
    main()
