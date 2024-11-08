import logging
import sys

import click

from ragamuffin.cli.utils import format_list
from ragamuffin.libraries.local import LocalLibrary
from ragamuffin.libraries.zotero import ZoteroLibrary
from ragamuffin.models.select import configure_llamaindex_embedding_model, get_llm_by_name
from ragamuffin.settings import get_settings
from ragamuffin.storage.utils import get_storage

logger = logging.getLogger(__name__)


@click.command()
@click.option("--generate", is_flag=True, help="Generate the RAG index.")
@click.option("--collection", multiple=True, help="Zotero collections to include when generating the index.")
@click.option("--name", help="Name of the Zotero chat agent. Default is 'zoterochat'.", default="zoterochat")
@click.version_option(message="Ragamuffin %(version)s")
def zotero_chat(generate: bool, collection: list[str], name: str) -> None:
    """Start the chat interface."""
    logger.info("Starting Zotero chat...")
    settings = get_settings()
    storage = get_storage()

    if generate:
        library = ZoteroLibrary(
            library_id=settings["zotero_library_id"], api_key=settings["zotero_api_key"], collections=collection
        )
        reader = library.get_reader()

        logger.info("Generating RAG embeddings...")
        index = storage.generate_index(name, reader)

    else:
        active_agents = storage.list_agents()
        if name not in active_agents:
            logger.error(f"Agent '{name}' not found.")
            logger.info("Run 'zotero-chat --generate' to create a new chat agent base on your library.")
            sys.exit(3)

        logger.info("Loading the RAG embedding index...")
        configure_llamaindex_embedding_model()
        index = storage.load_index(name)

    logger.info("Starting the chat interface...")
    llm = get_llm_by_name(settings.get("llm_model"))
    agent = index.as_chat_engine(llm=llm, similarity_top_k=6)
    from ragamuffin.webui.gradio_chat import GradioAgentChatUI

    webapp = GradioAgentChatUI(agent, name="Zotero")
    webapp.run()


@click.group(help="Ragamuffin RAG Chat Agents.")
@click.version_option(message="Ragamuffin %(version)s")
def cli() -> None:
    """Muffin CLI."""


@cli.command(name="generate")
@click.argument("name")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
def create_agent(name: str, source_dir: str) -> None:
    """Create a new chat agent using a directory of documents.

    \b
    Args:
        name: A name for the chat agent.
        source_dir: A directory containing the documents it will know.
    """
    logger.info(f"Creating a new chat agent '{name}' from '{source_dir}'.")
    storage = get_storage()

    library = LocalLibrary(library_dir=source_dir)
    reader = library.get_reader()

    logger.info("Generating RAG embeddings...")
    storage.generate_index(name, reader)

    logger.info(f"Agent '{name}' created successfully.")


@cli.command
@click.argument("name")
def chat(name: str) -> None:
    """Start a chat agent."""
    logger.info(f"Starting the chat interface for agent '{name}'.")
    storage = get_storage()
    settings = get_settings()

    # Check if the agent exists
    active_agents = storage.list_agents()
    if name not in active_agents:
        logger.error(f"Agent '{name}' not found.")
        logger.info(f"Available chat agents:\n\n{format_list(active_agents)}", extra={"markup": True})
        sys.exit(1)

    logger.info("Loading the RAG embedding index...")
    configure_llamaindex_embedding_model()
    index = storage.load_index(name)

    logger.info("Starting the chat interface...")

    llm = get_llm_by_name(settings.get("llm_model"))
    agent = index.as_chat_engine(llm=llm, similarity_top_k=6)
    from ragamuffin.webui.gradio_chat import GradioAgentChatUI

    webapp = GradioAgentChatUI(agent, name=name)
    webapp.run()


@cli.command
def agents() -> None:
    """List the available chat agents."""
    storage = get_storage()
    active_agents = storage.list_agents()

    if not active_agents:
        logger.info("No chat agents available. Use 'muffin generate' to create one.")
        return

    logger.info(f"Available chat agents:\n\n{format_list(active_agents)}", extra={"markup": True})


@cli.command
@click.argument("name")
def delete(name: str) -> None:
    """Delete a chat agent.

    \b
    Args:
        name: The name of the chat agent to delete. Use 'muffin agents' to list them.
    """
    storage = get_storage()
    storage.delete_agent(name)


if __name__ == "__main__":
    cli()
