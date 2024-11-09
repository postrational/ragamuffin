import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import default_file_metadata_func
from pyzotero.zotero import Zotero

from ragamuffin.cli.utils import format_list, track
from ragamuffin.libraries.interface import Library
from ragamuffin.libraries.utils import extract_year

logger = logging.getLogger(__name__)


class ZoteroLibrary(Library):
    def __init__(self, library_id: str, api_key: str, collections: list[str] | None = None):
        self.library_id = library_id
        self.api_key = api_key
        self.storage_dir = Path(tempfile.gettempdir()) / "ragamuffin" / "zotero"
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.articles: dict[str, dict] = {}
        self.zot = Zotero(
            library_id=self.library_id,
            library_type="user",
            api_key=self.api_key,
        )
        self.collections = self.get_selected_collections(collections) if collections else None

    def get_reader(self) -> BaseReader:
        """Get a Llama Index reader for the downloaded files."""
        self.download_articles()
        input_files = self.get_files()
        if not input_files:
            logger.error("No articles were downloaded.")
            sys.exit(2)
        return SimpleDirectoryReader(input_files=input_files, file_metadata=self.get_file_metadata)

    def download_articles(self) -> None:
        """Download articles from the Zotero library."""
        self.storage_dir.mkdir(exist_ok=True)

        logger.info("Retrieving your Zotero library. This may take a few minutes...")

        if self.collections:
            items = []
            for collection_key in self.collections:
                collection_items = self.zot.everything(self.zot.collection_items_top(collection_key))
                items.extend(collection_items)
            logger.info(f"Total items in selected collections: {len(items)}")
        else:
            items = self.zot.everything(self.zot.top())
            logger.info(f"Total items: {len(items)}")

        logger.info("Downloading articles...")

        for item in track(items, description="Downloading..."):
            # Parse the article data
            article_metadata = self.parse_article_data(item)
            name = article_metadata["name"]
            attachment_key = article_metadata["attachment_key"]
            attachment_size = article_metadata["attachment_size"]
            url = article_metadata["url"]

            if not attachment_key:
                logger.warning(f"Skipping, no PDF attachment found: {name}")
                logger.info(f"Zotero URL: {url}")
                continue

            filename = self.storage_dir / f"{name}.pdf"
            self.articles[str(filename)] = article_metadata

            if filename.exists() and attachment_size == filename.stat().st_size:
                logger.info(f"Already downloaded: {name}")
                continue

            # Download the file
            with filename.open("wb") as f:
                f.write(self.zot.file(attachment_key))
            logger.info(f"Downloaded: {name}")

    @staticmethod
    def get_pdf_attachment(zotero_item: dict[str, Any]) -> dict[str, Any] | None:
        """Find a PDF attachment in a Zotero item."""
        if "attachment" in zotero_item["links"]:
            attachment = zotero_item["links"]["attachment"]
            if attachment["attachmentType"] == "application/pdf":
                return attachment
        return None

    @staticmethod
    def parse_article_data(zotero_item: dict[str, Any]) -> dict[str, str | None]:
        """Retrieve relevant article data from the Zotero item."""
        key = zotero_item["key"]
        item_data = zotero_item["data"]

        authors = item_data.get("creators", [])
        if len(authors) == 0:
            author_str = "Unknown"
        else:
            first_author_name = authors[0].get("lastName", authors[0].get("name", "Unknown"))
            author_str = first_author_name if len(authors) == 1 else f"{first_author_name} et al."

        publication_year = extract_year(item_data.get("date"))
        title = item_data.get("title")

        # Generate name
        name = f"({author_str}, {publication_year}) {title}" if publication_year else f"({author_str}) {title}"

        # Extract attachment if type is application/pdf
        attachment = ZoteroLibrary.get_pdf_attachment(zotero_item)
        attachment_key = attachment["href"].split("/")[-1] if attachment else None
        attachment_size = attachment["attachmentSize"] if attachment else None

        user_id = zotero_item["library"]["name"]
        url = f"https://www.zotero.org/{user_id}/items/{key}"

        return {
            "key": key,
            "name": name,
            "attachment_key": attachment_key,
            "attachment_size": attachment_size,
            "url": url,
        }

    def get_file_metadata(self, file_path: str) -> dict[str, Any]:
        """Get metadata for a file."""
        metadata = default_file_metadata_func(file_path)

        # Add additional metadata
        metadata["source"] = "Zotero"
        metadata.update(self.articles.get(file_path, {}))

        # Return not null value
        return {meta_key: meta_value for meta_key, meta_value in metadata.items() if meta_value is not None}

    def get_files(self) -> list[str]:
        """Get the list of downloaded files."""
        return list(self.articles.keys())

    def get_selected_collections(self, collections: list[str]) -> dict[str, str]:
        """Get the Zotero IDs of the selected collections."""
        if len(collections) > 1:
            logger.info(f"Selected collections: {collections}")
        else:
            logger.info(f"Selected collection: {collections[0]}")
        available_collections = self.fetch_user_collections()
        available_collections_names = sorted(available_collections.values())

        if any(collection not in available_collections_names for collection in collections):
            logger.error("One or more selected collections not found.")
            logger.info(f"Available collections:\n{format_list(available_collections_names)}", extra={"markup": True})
            sys.exit(2)

        return {key: name for key, name in available_collections.items() if name in collections}

    def fetch_user_collections(self) -> dict[str, str]:
        """Fetch all collections from the user's Zotero library."""
        logger.info("Fetching your Zotero collections... This may take a few minutes.")
        collections = self.zot.all_collections()
        logger.info(f"Found {len(collections)} collections.")
        return {collection["key"]: collection["data"]["name"] for collection in collections}
