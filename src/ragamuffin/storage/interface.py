from abc import ABC, abstractmethod

from llama_index.core.indices.base import BaseIndex
from llama_index.core.readers.base import BaseReader


class Storage(ABC):
    @abstractmethod
    def generate_index(self, agent_name: str, reader: BaseReader) -> BaseIndex:
        """Load the documents and create a RAG index."""

    @abstractmethod
    def load_index(self, agent_name: str) -> BaseIndex:
        """Load the index from storage."""

    @abstractmethod
    def list_agents(self) -> list[str]:
        """Get the list of agents."""

    @abstractmethod
    def delete_agent(self, agent_name: str) -> None:
        """Delete the agent from storage."""
