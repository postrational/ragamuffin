from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI


def get_llm_by_name(name: str) -> LLM:
    """Get the LLM model by name."""
    provider, model_name = name.split("/", 1)
    provider = provider.lower()

    if provider == "openai":
        return OpenAI(model=model_name)

    raise ValueError(f"Unsupported LLM provider: {provider}")
