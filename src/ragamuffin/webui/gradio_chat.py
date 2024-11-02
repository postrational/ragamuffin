from pathlib import Path
from typing import Any

import gradio as gr
from gradio.themes.utils import colors, fonts
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llama_pack import BaseLlamaPack

from ragamuffin.nlp.highlighter import SemanticHighlighter


class GradioAgentChatUI(BaseLlamaPack):
    def __init__(
        self,
        agent: BaseChatEngine,
        **kwargs: dict,
    ) -> None:
        """Init params."""
        self.agent = agent
        self.semantic_highlighter = SemanticHighlighter()

    def get_modules(self) -> dict[str, Any]:
        """Get modules."""
        return {"agent": self.agent}

    def run(self, *args: list, **kwargs: dict) -> None:
        """Run the pipeline."""
        rafamuffin_theme = gr.themes.Soft(
            primary_hue=colors.cyan,
            secondary_hue=colors.rose,
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
        css_filename = Path(__file__).parent / "styles.css"

        webui = gr.Blocks(
            title="Ragamuffin Chat",
            theme=rafamuffin_theme,
            css_paths=[css_filename],
        )
        with webui:
            with gr.Row():
                # Left Column
                with gr.Column(scale=3):
                    gr.Markdown("### Ragamuffin Zotero Chat 🦙")
                    chat_window = gr.Chatbot(label="Conversation")
                    message = gr.Textbox(label="Write A Message")
                    clear = gr.ClearButton()

                # Right Column
                with gr.Column(scale=1):
                    console = gr.HTML(elem_id="sources")

            message.submit(
                self.handle_user_message,
                [message, chat_window],
                [message, chat_window],
                queue=False,
            ).then(
                self.generate_response,
                chat_window,
                [chat_window, console],
            )
            clear.click(self.reset_chat, None, [message, chat_window, console])

        webui.launch(server_port=8080, share=False)

    def handle_user_message(self, user_message: str, history: list[str]) -> tuple[str, list[str]]:
        """Handle the user submitted message. Clear message box, and append to the history."""
        return "", [*history, (user_message, "")]

    def generate_response(self, chat_history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
        """Generate a response from the agent."""
        query = chat_history[-1][0]
        response = self.agent.stream_chat(query)

        output = []
        for source in response.sources:
            for node_with_score in source.raw_output.source_nodes:
                text_node = node_with_score.node
                filename = text_node.metadata.get("file_name", "Unknown Filename")
                page = text_node.metadata.get("page_label")
                text = text_node.text
                selected_text = self.semantic_highlighter.highlight_text(query, text)
                page = f"<br>Page {page}" if page else ""
                output.append(f"<p><b>{filename}</b><br>{selected_text}{page}</p>")

        html_output = "".join(output)

        for token in response.response_gen:
            chat_history[-1][1] += token
            yield chat_history, str(html_output)

    def reset_chat(self) -> tuple[str, str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        self.agent.reset()  # clear agent history
        return "", "", ""
