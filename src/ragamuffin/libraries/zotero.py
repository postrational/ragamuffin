import logging
import tempfile
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import default_file_metadata_func
from pyzotero.zotero import Zotero

from ragamuffin.libraries.interface import Library
from ragamuffin.libraries.utils import extract_year
from ragamuffin.progress_bar import track
from ragamuffin.settings import get_settings

logger = logging.getLogger(__name__)


class ZoteroLibrary(Library):
    def __init__(self, library_id: str, api_key: str):
        self.library_id = library_id
        self.api_key = api_key
        self.storage_dir = Path(tempfile.gettempdir()) / "ragamuffin" / "zotero"
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.articles = {}

    def get_reader(self) -> BaseReader:
        """Get a Llama Index reader for the downloaded files."""
        self.download_all_articles()
        file_metadata = self.get_file_metadata
        input_files = self.get_files()
        return SimpleDirectoryReader(input_files=input_files, file_metadata=file_metadata)

    def download_all_articles(self) -> None:
        """Download all articles from the Zotero library."""
        self.storage_dir.mkdir(exist_ok=True)

        zot = Zotero(
            library_id=self.library_id,
            library_type="user",
            api_key=self.api_key,
        )

        logger.info("Retrieving your Zotero library...")
        items = zot.everything(zot.top())
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
                f.write(zot.file(attachment_key))
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

        authors = zotero_item["data"]["creators"]
        if len(authors) == 0:
            author_str = "Unknown"
        elif len(authors) == 1:
            author_str = authors[0]["lastName"]
        else:
            author_str = f"{authors[0]['lastName']} et al."

        publication_year = extract_year(zotero_item["data"]["date"])
        title = zotero_item["data"]["title"]

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


if __name__ == "__main__":
    settings = get_settings()

    library = ZoteroLibrary(library_id=settings["zotero_library_id"], api_key=settings["zotero_api_key"])
    library.download_all_articles()
    reader = library.get_reader()
    logger.info("Reading the downloaded files...")
    data_iter = reader.iter_data()
    logger.info("Example data:")
    logger.info(next(data_iter))
