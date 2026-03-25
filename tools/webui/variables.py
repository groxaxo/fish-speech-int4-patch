from fish_speech.i18n import i18n

HEADER_MD = """
<div class="fs-hero">
  <div class="fs-pill">Fish Speech S2-Pro - BnB NF4 - OpenAI-Compatible</div>
  <h1>Fish Speech S2-Pro, tuned for 12 GB GPUs</h1>
  <p>
    A polished voice studio for Groxaxo's Fish Speech fork: OpenAI-style TTS,
    reference-audio cloning, and default BnB4 startup for practical single-GPU deployment.
  </p>
  <div class="fs-links">
    <a href="https://github.com/groxaxo/fish-speech-int4-patch" target="_blank">GitHub</a>
    <a href="https://huggingface.co/groxaxo/s2-pro" target="_blank">S2-Pro checkpoint</a>
    <a href="https://fish.audio/blog/fish-audio-open-sources-s2/" target="_blank">Tech blog</a>
  </div>
</div>
"""

TEXTBOX_PLACEHOLDER = i18n(
    "Write narration, dialogue, or inline emotion tags here."
)

PROMPT_EXAMPLES = [
    [
        "Narrate this in a calm documentary voice: The storm rolled over the valley at dawn, leaving the city wrapped in silver light."
    ],
    [
        "[warm and reassuring] Thanks for calling. Your order has shipped and will arrive tomorrow afternoon."
    ],
    [
        "<|speaker:0|> [excited] We did it! <|speaker:1|> [gentle laugh] I told you this plan would work."
    ],
]

REFERENCE_TIPS_MD = f"""
### {i18n("Reference guidance")}

- {i18n("Use 5 to 10 seconds of clean speech for the strongest cloning result.")}
- {i18n("Provide matching transcript text for the uploaded reference clip.")}
- {i18n("OpenAI-style voices like alloy or nova use the default model voice path.")}
- {i18n("A saved reference ID lets you reuse a voice without uploading audio again.")}
"""

API_COMPAT_MD = """
### API-ready defaults

- `POST /v1/audio/speech`
- canonical model: `s2-pro`
- compatible model IDs: `tts-1`, `tts-1-hd`, `fish-speech`
- defaults: BnB4, FP16, lazy API loading, `:8880`
"""

APP_CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
    radial-gradient(circle at top right, rgba(99, 102, 241, 0.16), transparent 26%),
    linear-gradient(180deg, #0f172a 0%, #111827 35%, #0b1120 100%);
}

.fs-hero {
  padding: 28px 30px;
  margin-bottom: 18px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.88));
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28);
}

.fs-hero h1,
.fs-hero p,
.fs-hero a,
.fs-pill {
  color: #e5eefc !important;
}

.fs-hero h1 {
  margin: 12px 0 10px !important;
  font-size: 2.1rem !important;
}

.fs-hero p {
  max-width: 780px;
  margin: 0 0 18px !important;
  line-height: 1.6;
}

.fs-pill {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(14, 165, 233, 0.16);
  border: 1px solid rgba(125, 211, 252, 0.3);
  font-size: 0.88rem;
  font-weight: 600;
}

.fs-links {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.fs-links a {
  text-decoration: none;
  padding: 8px 14px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.16);
  border: 1px solid rgba(96, 165, 250, 0.26);
}

.fs-card {
  height: 100%;
  border-radius: 20px;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(15, 23, 42, 0.68);
  padding: 18px 18px 4px;
}

.fs-card h3,
.fs-card p,
.fs-card li {
  color: #dbe7ff !important;
}

.fs-output audio {
  width: 100%;
}
"""
