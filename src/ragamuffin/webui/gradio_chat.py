from collections.abc import Callable
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
                    with gr.Row():
                        clear = gr.ClearButton()
                        submit = gr.Button("Submit", variant="primary")

                # Right Column
                with gr.Column(scale=1):
                    console = gr.HTML(elem_id="sources")

            def apply_submit_action(component_action: Callable) -> None:
                component_action(
                    self.handle_user_message,
                    inputs=[message, chat_window],
                    outputs=[message, chat_window],
                    queue=False,
                ).then(
                    self.generate_response,
                    inputs=chat_window,
                    outputs=[chat_window, console],
                )

            # Apply actions
            apply_submit_action(message.submit)
            apply_submit_action(submit.click)
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
                metadata = text_node.metadata
                page = metadata.get("page_label")

                filename = metadata.get("file_name", "Unknown Filename")
                name = metadata.get("name", filename)
                url = metadata.get("url")
                filename = f"<a href='{url}' target='_blank'>{name}</a>" if url else f"<b>{name}</b>"

                selected_text = self.semantic_highlighter.highlight_text(query, text_node.text)
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
