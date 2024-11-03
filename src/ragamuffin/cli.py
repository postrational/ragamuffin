import logging

import click

from ragamuffin.libraries.local import LocalLibrary
from ragamuffin.libraries.zotero import ZoteroLibrary
from ragamuffin.settings import get_settings
from ragamuffin.storage.cassandra import CassandraStorage

logger = logging.getLogger(__name__)


@click.command()
@click.option("--generate", is_flag=True, help="Generate the RAG index.")
def zotero_chat(generate: bool) -> None:
    """Start the chat interface."""
    logger.info("Starting Zotero chat...")
    settings = get_settings()

    logger.info("Connecting to the Cassandra cluster...")
    storage = CassandraStorage(
        cluster_ip=settings.get("cassandra_cluster_ip"), keyspace=settings.get("cassandra_keyspace"), table="zoterochat"
    )

    if generate:
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
    agent = index.as_chat_engine(similarity_top_k=6)
    from ragamuffin.webui.gradio_chat import GradioAgentChatUI

    webapp = GradioAgentChatUI(agent, name="Zotero")
    webapp.run()


@click.group(help="Ragamuffin RAG Chat Agents.")
def cli() -> None:
    """Muffin CLI."""


@cli.command(name="generate", help="Generate the RAG index for a chat agent.")
@click.argument("name")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
def create_agent(name: str, source_dir: str) -> None:
    """Create a new chat agent using a directory of documents."""
    logger.info(f"Creating a new chat agent '{name}' from '{source_dir}'.")
    settings = get_settings()

    logger.info("Connecting to the Cassandra cluster...")
    storage = CassandraStorage(
        cluster_ip=settings.get("cassandra_cluster_ip"), keyspace=settings.get("cassandra_keyspace"), table=name
    )

    library = LocalLibrary(library_dir=source_dir)
    reader = library.get_reader()

    logger.info("Generating RAG embeddings...")
    storage.generate_index(reader)

    logger.info(f"Agent '{name}' created successfully.")


@cli.command(name="chat")
@click.argument("name")
def chat(name: str) -> None:
    """Start a chat agent."""
    logger.info(f"Starting the chat interface for agent '{name}'.")
    settings = get_settings()

    logger.info("Connecting to the Cassandra cluster...")
    storage = CassandraStorage(
        cluster_ip=settings.get("cassandra_cluster_ip"), keyspace=settings.get("cassandra_keyspace"), table=name
    )

    logger.info("Loading the RAG embedding index...")
    index = storage.load_index()

    logger.info("Starting the chat interface...")
    agent = index.as_chat_engine(similarity_top_k=6)
    from ragamuffin.webui.gradio_chat import GradioAgentChatUI

    webapp = GradioAgentChatUI(agent, name=name)
    webapp.run()


if __name__ == "__main__":
    cli()
