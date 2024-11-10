from ragamuffin.error_handling import ensure_string
from ragamuffin.models.model_picker import get_llm_by_name
from ragamuffin.settings import get_settings


class QueryEnhancer:
    def __init__(self):
        settings = get_settings()
        llm_model = ensure_string(settings.get("llm_model"))
        self.model = get_llm_by_name(llm_model)

    def __call__(self, chat_history: list[dict]) -> str:
        """Enhance the last query in the chat history."""
        return self.enhance(chat_history)

    def enhance(self, chat_history: list[dict]) -> str:
        """Enhance the last query in the chat history."""
        context_str = "\n".join(
            [f"{idx + 1}. {msg['content']}" for idx, msg in enumerate(chat_history) if msg["role"] == "user"]
        )
        query_str = chat_history[-1]["content"]
        prompt = (
            "You are an expert Q&A system that is trusted around the world.\n"
            "The user has provided a query and wants to search for matching sources.\n"
            "Please rewrite the query and add keywords to improve the changes of finding relevant sources.\n"
            "The search is based on semantic similarity as part of a RAG-based system.\n"
            "There are some rule: \n"
            "1. Only return the enhanced query, no other output. \n"
            "2. Focus on the meaning of the last query and don't mix in previous queries unless it's needed. \n"
            "All queries in the conversation:\n"
            "---------------------\n"
            f"{context_str}\n"
            "---------------------\n"
            "The query to enhance is:\n"
            f"{query_str}\n"
        )
        response = self.model.complete(prompt)
        return response.text
