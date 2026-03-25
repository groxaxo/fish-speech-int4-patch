from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui.variables import (
    API_COMPAT_MD,
    APP_CSS,
    HEADER_MD,
    PROMPT_EXAMPLES,
    REFERENCE_TIPS_MD,
    TEXTBOX_PLACEHOLDER,
)


def _build_theme():
    return gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="rgba(15, 23, 42, 0.72)",
        block_border_color="rgba(148, 163, 184, 0.14)",
        button_primary_background_fill="linear-gradient(90deg, #38bdf8, #818cf8)",
        button_primary_border_color="transparent",
    )


def build_app(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    theme_object = _build_theme()

    with gr.Blocks(title="Fish Speech S2-Pro") as app:
        gr.HTML(HEADER_MD)

        if theme in {"light", "dark"}:
            app.load(
                None,
                None,
                js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
                % theme,
            )

        with gr.Row(equal_height=True):
            with gr.Column(elem_classes=["fs-card"]):
                gr.Markdown(
                    "### 12 GB friendly\n"
                    "BnB NF4 is the default path, so S2-Pro is practical on RTX 3060-class GPUs."
                )
            with gr.Column(elem_classes=["fs-card"]):
                gr.Markdown(
                    "### OpenAI-style serving\n"
                    "Use the same repo for `/v1/audio/speech`, model discovery, and reference-aware TTS."
                )
            with gr.Column(elem_classes=["fs-card"]):
                gr.Markdown(
                    "### Voice cloning workflow\n"
                    "Upload a short reference clip, add its transcript, then generate expressive audio in one pass."
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=7, elem_classes=["fs-card"]):
                text = gr.Textbox(
                    label=i18n("Input Text"),
                    placeholder=TEXTBOX_PLACEHOLDER,
                    lines=12,
                )
                gr.Examples(
                    examples=PROMPT_EXAMPLES,
                    inputs=[text],
                    label=i18n("Prompt ideas"),
                )

                with gr.Row():
                    generate = gr.Button(
                        value="\U0001f3a7 " + i18n("Generate"),
                        variant="primary",
                    )
                    gr.ClearButton(
                        [text],
                        value=i18n("Clear text"),
                    )

            with gr.Column(scale=5, elem_classes=["fs-card", "fs-output"]):
                audio = gr.Audio(
                    label=i18n("Generated Audio"),
                    type="numpy",
                    interactive=False,
                    visible=True,
                )
                error = gr.HTML(
                    value="",
                    label=i18n("Error Message"),
                    visible=True,
                )
                gr.Markdown(API_COMPAT_MD)

        with gr.Row(equal_height=True):
            with gr.Column(scale=4, elem_classes=["fs-card"]):
                with gr.Accordion(label=i18n("Reference Audio"), open=True):
                    gr.Markdown(
                        i18n(
                            "Upload a short reference clip with its transcript, or reuse a saved reference ID."
                        )
                    )
                    reference_id = gr.Textbox(
                        label=i18n("Reference ID"),
                        placeholder="Leave empty to use uploaded references",
                    )
                    use_memory_cache = gr.Radio(
                        label=i18n("Use Memory Cache"),
                        choices=["on", "off"],
                        value="on",
                    )
                    reference_audio = gr.Audio(
                        label=i18n("Reference Audio"),
                        type="filepath",
                    )
                    reference_text = gr.Textbox(
                        label=i18n("Reference Text"),
                        lines=2,
                        placeholder="A matching transcript for the uploaded reference clip.",
                        value="",
                    )

            with gr.Column(scale=4, elem_classes=["fs-card"]):
                with gr.Accordion(label=i18n("Advanced Config"), open=True):
                    chunk_length = gr.Slider(
                        label=i18n("Iterative Prompt Length"),
                        minimum=100,
                        maximum=300,
                        value=200,
                        step=10,
                    )

                    max_new_tokens = gr.Slider(
                        label=i18n("Maximum tokens per batch, 0 means no limit"),
                        minimum=0,
                        maximum=2048,
                        value=1024,
                        step=8,
                    )

                    top_p = gr.Slider(
                        label="Top-P",
                        minimum=0.7,
                        maximum=0.95,
                        value=0.8,
                        step=0.01,
                    )

                    repetition_penalty = gr.Slider(
                        label=i18n("Repetition Penalty"),
                        minimum=1.0,
                        maximum=1.5,
                        value=1.1,
                        step=0.01,
                    )

                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.7,
                        maximum=1.0,
                        value=0.8,
                        step=0.01,
                    )
                    seed = gr.Number(
                        label="Seed",
                        info="0 means randomized inference, otherwise deterministic",
                        value=0,
                    )

            with gr.Column(scale=4, elem_classes=["fs-card"]):
                gr.Markdown(REFERENCE_TIPS_MD)

        # Submit
        generate.click(
            inference_fct,
            [
                text,
                reference_id,
                reference_audio,
                reference_text,
                max_new_tokens,
                chunk_length,
                top_p,
                repetition_penalty,
                temperature,
                seed,
                use_memory_cache,
            ],
            [audio, error],
            concurrency_limit=1,
        )

    setattr(app, "_fs_theme", theme_object)
    setattr(app, "_fs_css", APP_CSS)
    return app
