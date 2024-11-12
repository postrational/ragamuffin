from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from ragamuffin.error_handling import ConfigurationError
from ragamuffin.settings import get_settings


def get_llm_by_name(name: str) -> LLM:
    """Get the LLM model by name."""
    provider, model_name = name.split("/", 1)
    provider = provider.lower()

    if provider == "openai":
        return OpenAI(model=model_name)

    raise ConfigurationError(f"Unsupported LLM provider: {provider}")


def get_embedding_model_by_name(name: str) -> BaseEmbedding:
    """Get the Hugging Face embedding model by name."""
    try:
        provider, model_name = name.split("/", 1)
        provider = provider.lower()
    except ValueError as e:
        raise ConfigurationError(f"Unrecognized embedding model name: {name}") from e

    if provider == "huggingface.co":
        return HuggingFaceEmbedding(model_name=model_name)

    if provider == "openai":
        return OpenAIEmbedding(model=model_name)

    raise ConfigurationError(f"Unsupported embedding provider: {provider}")


def configure_llamaindex_embedding_model() -> None:
    """Configure the LlamaIndex embeddings for RAG."""
    settings = get_settings()
    # Configure chunking settings
    Settings.chunk_size = 256
    Settings.chunk_overlap = 48
    # Set the embedding model
    Settings.embed_model = get_embedding_model_by_name(str(settings.get("embedding_model")))
