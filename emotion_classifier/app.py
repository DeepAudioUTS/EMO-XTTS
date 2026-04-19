"""
app.py

Gradio demo for the 8-class emotion classifier.

Run with:
    python app.py

Opens a local Gradio UI at http://127.0.0.1:7860
"""

from __future__ import annotations

import html
from typing import Any

import gradio as gr

from data_pipeline import EMOTION_LABELS
from inference import chunk_and_predict, load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH: str = "./emotion_model"
SERVER_PORT: int = 7860
SERVER_NAME: str = "127.0.0.1"

# Emotion → highlight colour (CSS background-color values)
EMOTION_COLORS: dict[str, str] = {
    "Neutral": "#d3d3d3",       # grey
    "Happy": "#ffe066",         # yellow
    "Sad": "#6699cc",           # blue
    "Tense/Fearful": "#9b59b6", # purple
    "Angry": "#e74c3c",         # red
    "Surprised": "#f39c12",     # orange
    "Disgusted": "#27ae60",     # green
    "Loving/Tender": "#f1a7c1", # pink
}

# ---------------------------------------------------------------------------
# Global model (loaded once on startup)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None


def _ensure_model_loaded() -> tuple[Any, Any]:
    """Lazy-load the model on first call."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_model(DEFAULT_MODEL_PATH)
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _highlight_tokens(chunk_text: str, emotion_spans: list[dict], emotion: str) -> str:
    """
    Build an HTML string where the top emotion-driving tokens are
    highlighted with the emotion's background colour.

    Args:
        chunk_text:    Raw text of the chunk.
        emotion_spans: List of span dicts from ``predict()``.
        emotion:       Emotion label string for colour lookup.

    Returns:
        HTML string with ``<mark>`` tags around matched tokens.
    """
    color = EMOTION_COLORS.get(emotion, "#eeeeee")
    highlighted_words = {span["word"].lower() for span in emotion_spans}

    # Simple word-by-word replacement (works for whole words; sub-words merged)
    words = chunk_text.split()
    result_parts: list[str] = []
    for word in words:
        # Strip punctuation for matching
        clean = word.strip(".,!?;:\"'()[]").lower()
        if clean in highlighted_words:
            safe = html.escape(word)
            result_parts.append(
                f'<mark style="background-color:{color};padding:2px 4px;'
                f'border-radius:3px;">{safe}</mark>'
            )
        else:
            result_parts.append(html.escape(word))
    return " ".join(result_parts)


def _build_highlighted_html(result: dict[str, Any]) -> str:
    """
    Render the full highlighted HTML for all chunks.

    Args:
        result: Output dict from ``chunk_and_predict()``.

    Returns:
        HTML string suitable for ``gr.HTML``.
    """
    lines: list[str] = ['<div style="font-family:sans-serif;line-height:1.8;">']
    for chunk_info in result.get("chunks", []):
        emotion = chunk_info["label"]
        color = EMOTION_COLORS.get(emotion, "#eeeeee")
        spans = chunk_info.get("emotion_spans", [])
        chunk_html = _highlight_tokens(chunk_info["chunk"], spans, emotion)
        lines.append(
            f'<p><span style="background:{color};padding:2px 6px;'
            f'border-radius:4px;font-weight:bold;font-size:0.85em;">'
            f"{html.escape(emotion)}</span> "
            f"{chunk_html}</p>"
        )
    lines.append("</div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------


def classify_text(input_text: str) -> tuple[str, list[list[str]], str]:
    """
    Main Gradio callback.  Runs chunk_and_predict and formats three outputs.

    Args:
        input_text: Raw user-entered text.

    Returns:
        Tuple of:
          - Overall emotion summary string
          - Per-chunk table rows (list of [chunk, label, confidence])
          - Highlighted HTML string
    """
    if not input_text or not input_text.strip():
        return "No input provided.", [], "<p>Enter some text above.</p>"

    try:
        model, tokenizer = _ensure_model_loaded()
        result = chunk_and_predict(input_text.strip(), model, tokenizer)
    except FileNotFoundError as exc:
        return str(exc), [], f"<p style='color:red;'>{html.escape(str(exc))}</p>"
    except Exception as exc:
        return f"Error: {exc}", [], f"<p style='color:red;'>{html.escape(str(exc))}</p>"

    overall = (
        f"{result['overall_emotion']} "
        f"(confidence: {result['overall_confidence']:.1%})"
    )

    table_rows = [
        [
            c["chunk"],
            c["label"],
            f"{c['confidence']:.1%}",
        ]
        for c in result.get("chunks", [])
    ]

    highlighted_html = _build_highlighted_html(result)
    return overall, table_rows, highlighted_html


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    with gr.Blocks(title="Emotion Classifier") as demo:
        gr.Markdown(
            "## Emotion Classifier\n"
            "Enter any multi-sentence text. The model will classify emotions "
            "sentence-by-sentence and highlight the key tokens."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Paste a passage, dialogue, or any text here...",
                    lines=8,
                )
                submit_btn = gr.Button("Classify", variant="primary")

        with gr.Row():
            overall_output = gr.Textbox(
                label="Overall Emotion",
                interactive=False,
            )

        with gr.Row():
            chunk_table = gr.Dataframe(
                headers=["Chunk", "Label", "Confidence"],
                label="Per-Chunk Breakdown",
                wrap=True,
                interactive=False,
            )

        with gr.Row():
            highlighted_html = gr.HTML(label="Highlighted Text")

        # Legend
        legend_html = "".join(
            f'<span style="background:{color};padding:3px 8px;margin:2px;'
            f'border-radius:4px;">{html.escape(emotion)}</span> '
            for emotion, color in EMOTION_COLORS.items()
        )
        gr.HTML(f"<p><b>Colour legend:</b> {legend_html}</p>")

        submit_btn.click(
            fn=classify_text,
            inputs=[text_input],
            outputs=[overall_output, chunk_table, highlighted_html],
        )

        # Example inputs
        gr.Examples(
            examples=[
                [
                    "She walked slowly into the empty room. "
                    "Tears streamed down her face as she remembered him. "
                    "He was gone, and nothing would ever be the same."
                ],
                [
                    "Congratulations! You just won the championship! "
                    "The crowd erupted in joy. Everyone was hugging each other."
                ],
                [
                    "He slammed the door and shouted at the top of his lungs. "
                    "How dare they treat him like this!"
                ],
            ],
            inputs=[text_input],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=False,
    )
