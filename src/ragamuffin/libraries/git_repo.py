import logging
import re
import shutil
import tempfile
from pathlib import Path

from git import GitCommandError, Repo
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from typing_extensions import override

from ragamuffin.libraries.interface import Library

logger = logging.getLogger(__name__)


class GitLibrary(Library):
    def __init__(self, git_repo: str, ref: str | None = None, **kwargs: dict):
        super().__init__(**kwargs)
        self.git_repo = git_repo
        self.ref = ref

        # Create a unique storage directory for the cloned repository
        repo_slug = re.sub(r"[^\w\-]", "_", git_repo)
        repo_slug += f"_{ref}" if ref else ""
        self.storage_dir = Path(tempfile.gettempdir()) / "ragamuffin" / "git" / repo_slug

    def download_repo(self, output_directory: Path, repo_url: str, ref: str | None = None) -> None:
        """Downloads a Git repository from a given URL and extracts its files to the specified output directory.

        Args:
            repo_url: The HTTPS or SSH URL of the Git repository.
            output_directory: The path to the directory where the repository files will be saved.
            ref: The branch, tag, or commit hash to checkout. Defaults to None (uses default branch).

        """
        logger.info(f"Cloning Git repository from {repo_url}")

        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        if ref is None:
            Repo.clone_from(repo_url, output_directory, depth=1)
        else:
            try:
                Repo.clone_from(repo_url, output_directory, depth=1, branch=ref, single_branch=True)
            except GitCommandError:
                # If cloning with depth=1 fails (e.g., for commit hash), perform a full clone
                repo = Repo.clone_from(repo_url, output_directory)
                repo.git.checkout(ref)

        # Remove the .git directory to exclude repository metadata
        git_dir = output_directory / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)

    @override
    def get_reader(self) -> BaseReader:
        """Get a Llama Index reader for the cloned repository."""
        self.download_repo(self.storage_dir, self.git_repo, self.ref)
        return SimpleDirectoryReader(input_dir=self.storage_dir, recursive=True)
