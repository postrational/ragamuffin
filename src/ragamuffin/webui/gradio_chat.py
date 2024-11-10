import html
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import gradio as gr
from gradio.themes.utils import colors, fonts
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core.schema import NodeWithScore

from ragamuffin.models.enhancer import QueryEnhancer
from ragamuffin.models.highlighter import SemanticHighlighter


class GradioAgentChatUI(BaseLlamaPack):
    def __init__(
        self,
        agent: BaseChatEngine,
        *,
        name: str = "Unnamed",
        **kwargs: dict,
    ):
        """Init params."""
        self.agent = agent
        self.semantic_highlighter = SemanticHighlighter()
        self.query_enhancer = QueryEnhancer()
        self.title = f"Ragamuffin {snake_to_title_case(name)} Chat"

    def get_modules(self) -> dict[str, Any]:
        """Get modules."""
        return {"agent": self.agent}

    def run(self, *args: list, **kwargs: dict) -> None:
        """Run the pipeline."""
        rafamuffin_theme = gr.themes.Soft(
            primary_hue=colors.cyan,
            secondary_hue=colors.sky,
            neutral_hue=colors.gray,
            font=(
                fonts.GoogleFont("Quicksand"),
                "ui-sans-serif",
                "sans-serif",
            ),
            font_mono=(
                fonts.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "monospace",
            ),
        )
        css_filename = Path(__file__).parent / "style.css"

        webui = gr.Blocks(
            title=self.title,
            theme=rafamuffin_theme,
            css_paths=[css_filename],
        )
        with webui:
            with gr.Row():
                # Left Column
                with gr.Column(scale=3):
                    gr.Markdown(f"### {self.title} 🐈")
                    chat_window = gr.Chatbot(label="Conversation", elem_id="chatbot", type="messages")
                    message = gr.Textbox(label="Write A Message")
                    with gr.Row():
                        clear = gr.ClearButton()
                        submit = gr.Button("Submit", variant="primary")

                # Right Column
                with gr.Column(scale=2):
                    console = gr.HTML(elem_id="sources")

            def apply_submit_action(component_action: Callable) -> None:
                component_action(
                    self.accept_message, inputs=[message, chat_window], outputs=[message, chat_window]
                ).then(self.respond, inputs=[chat_window], outputs=[chat_window, console])

            # Apply actions
            apply_submit_action(message.submit)
            apply_submit_action(submit.click)
            clear.click(self.reset_chat, None, [message, chat_window, console])

        webui.launch(inbrowser=True, share=False)

    def respond(self, chat_history: list[dict]) -> Generator[tuple[list[dict], str], None, None]:
        """Respond to the user message."""
        query = chat_history[-1]["content"]
        response = self.agent.stream_chat(query)

        sources_html = self.generate_sources_html(query, response.source_nodes)

        chat_history.append({"role": "assistant", "content": ""})
        for token in response.response_gen:
            chat_history[-1]["content"] += token
            yield chat_history, str(sources_html)

    def accept_message(self, user_message: str, chat_history: list[dict]) -> tuple[str, list[dict]]:
        """Accept the user message."""
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append(
            {
                "role": "assistant",
                "content": self.query_enhancer(chat_history),
                "metadata": {"title": "🧠 Building semantic search query"},
            }
        )
        return "", chat_history

    def reset_chat(self) -> tuple[str, str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        self.agent.reset()  # clear agent history
        return "", "", ""

    def generate_sources_html(self, query: str, source_nodes: list[NodeWithScore]) -> str:
        """Generate HTML for the sources."""
        output_html = ""
        sources_text = []
        nodes_info = []

        if not source_nodes:
            return "<p>No sources found.</p>"

        # Collect all texts and their associated metadata
        for node_with_score in source_nodes:
            text_node = node_with_score.node
            metadata = text_node.metadata
            score = node_with_score.score
            page = metadata.get("page_label")

            filename = metadata.get("file_name", "Unknown Filename")
            name = metadata.get("name", filename)
            url = metadata.get("url")
            filename_html = f"<a href='{url}' target='_blank'>{name}</a>" if url else f"<b>{name}</b>"

            # Append text and metadata to lists
            if text_node.text:
                sources_text.append(text_node.text)
                nodes_info.append(
                    {
                        "filename_html": filename_html,
                        "page": page,
                        "score": score,
                    }
                )

        # Highlight the texts
        sources_text = [html.escape(text) for text in sources_text]
        highlighted_texts = self.semantic_highlighter.highlight_multiple(query, sources_text)

        # Construct the output using the highlighted texts and metadata
        for highlighted_text, info in zip(highlighted_texts, nodes_info, strict=False):
            source_footer = f"<br>Page {info['page']}" if info["page"] else "<br>"
            similarity_class = int(min(score * 10, 9))
            source_footer += f" <span class='badge similarity-{similarity_class}'>{info['score']:.2f}</span>"
            output_html += f"<p><b>{info['filename_html']}</b><br>{highlighted_text}{source_footer}</p>"

        return output_html


def snake_to_title_case(snake_str: str) -> str:
    """Convert snake case to title case."""
    return snake_str.replace("_", " ").title()
