import io
from argparse import ArgumentParser
from http import HTTPStatus
from typing import Annotated, Any, Iterable

import numpy as np
import ormsgpack
import soundfile as sf
from baize.datastructures import ContentType
from kui.asgi import (
    HTTPException,
    HttpRequest,
    JSONResponse,
    request,
)
from loguru import logger
from pydantic import BaseModel

from fish_speech.text import normalize_text_for_tts
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import OpenAISpeechRequest, ServeTTSRequest
from tools.server.inference import inference_wrapper as inference

OPENAI_MODEL_METADATA = (
    {
        "id": "tts-1",
        "object": "model",
        "created": 1710000000,
        "owned_by": "fish-audio",
    },
    {
        "id": "tts-1-hd",
        "object": "model",
        "created": 1710000000,
        "owned_by": "fish-audio",
    },
    {
        "id": "fish-speech",
        "object": "model",
        "created": 1710000000,
        "owned_by": "fish-audio",
    },
    {
        "id": "s2-pro",
        "object": "model",
        "created": 1710000000,
        "owned_by": "fish-audio",
    },
)

OPENAI_VOICE_ALIASES = {
    "",
    "default",
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["tts"], default="tts")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Override model max_seq_len for KV-cache pre-allocation (saves VRAM on small GPUs)",
    )
    parser.add_argument("--listen", type=str, default="127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key", type=str, default=None)

    return parser.parse_args()


class MsgPackRequest(HttpRequest):
    async def data(
        self,
    ) -> Annotated[
        Any,
        ContentType("application/msgpack"),
        ContentType("application/json"),
        ContentType("multipart/form-data"),
    ]:
        if self.content_type == "application/msgpack":
            return ormsgpack.unpackb(await self.body)

        elif self.content_type == "application/json":
            return await self.json

        elif self.content_type == "multipart/form-data":
            return await self.form

        raise HTTPException(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            headers={
                "Accept": "application/msgpack, application/json, multipart/form-data"
            },
        )


async def inference_async(req: ServeTTSRequest, engine: TTSInferenceEngine):
    for chunk in inference(req, engine):
        if isinstance(chunk, bytes):
            yield chunk


async def buffer_to_async_generator(buffer):
    yield buffer


def get_content_type(audio_format):
    if audio_format == "wav":
        return "audio/wav"
    elif audio_format == "flac":
        return "audio/flac"
    elif audio_format == "mp3":
        return "audio/mpeg"
    elif audio_format == "pcm":
        return "audio/pcm"
    else:
        return "application/octet-stream"


def serialize_audio_output(audio: np.ndarray, sample_rate: int, audio_format: str) -> bytes:
    if audio_format == "pcm":
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=audio_format)
    return buffer.getvalue()


def chunk_bytes(data: bytes, chunk_size: int = 65536) -> Iterable[bytes]:
    for idx in range(0, len(data), chunk_size):
        yield data[idx : idx + chunk_size]


def build_openai_error(
    error: str, message: str, error_type: str = "invalid_request_error"
) -> dict[str, str]:
    return {"error": error, "message": message, "type": error_type}


def build_openai_model_list() -> dict[str, object]:
    return {"object": "list", "data": list(OPENAI_MODEL_METADATA)}


def get_openai_model(model_id: str) -> dict[str, object] | None:
    for model in OPENAI_MODEL_METADATA:
        if model["id"] == model_id:
            return model
    return None


def resolve_openai_reference_id(
    voice: str, available_reference_ids: list[str]
) -> str | None:
    normalized_voice = voice.strip()
    if normalized_voice.lower() in OPENAI_VOICE_ALIASES:
        return None

    if normalized_voice in available_reference_ids:
        return normalized_voice

    raise ValueError(
        f"Voice '{voice}' not found. Available reference IDs: {', '.join(sorted(available_reference_ids)) or 'none'}"
    )


def prepare_tts_request(req: ServeTTSRequest, max_text_length: int = 0) -> ServeTTSRequest:
    prepared = req.model_copy(deep=True)

    prepared.text = normalize_text_for_tts(
        prepared.text,
        normalize=prepared.normalize,
        normalization_options=prepared.normalization_options,
    )
    if not prepared.text:
        raise ValueError("Text is empty after preprocessing")

    if max_text_length > 0 and len(prepared.text) > max_text_length:
        raise ValueError(f"Text is too long, max length is {max_text_length}")

    if prepared.streaming and prepared.format != "wav":
        raise ValueError("Streaming only supports WAV format")

    normalized_refs = []
    for reference in prepared.references:
        normalized_text = normalize_text_for_tts(
            reference.text,
            normalize=prepared.normalize,
            normalization_options=prepared.normalization_options,
        )
        if not normalized_text:
            raise ValueError("Reference text is empty after preprocessing")
        normalized_refs.append(reference.model_copy(update={"text": normalized_text}))
    prepared.references = normalized_refs

    return prepared


def build_openai_tts_request(
    req: OpenAISpeechRequest, reference_id: str | None
) -> ServeTTSRequest:
    if req.speed != 1.0:
        raise ValueError(
            "Fish Speech does not currently support speed control via /v1/audio/speech; use speed=1.0"
        )

    return ServeTTSRequest(
        text=req.input,
        chunk_length=req.chunk_length,
        format=req.response_format,
        reference_id=reference_id,
        normalize=req.normalization_options.normalize,
        normalization_options=req.normalization_options,
        streaming=req.stream and req.response_format == "wav",
        max_new_tokens=req.max_new_tokens,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
    )


def wants_json(req):
    """Helper method to determine if the client wants a JSON response

    Parameters
    ----------
    req : Request
        The request object

    Returns
    -------
    bool
        True if the client wants a JSON response, False otherwise
    """
    q = req.query_params.get("format", "").strip().lower()
    if q in {"json", "application/json", "msgpack", "application/msgpack"}:
        return q == "json"
    accept = req.headers.get("Accept", "").strip().lower()
    return "application/json" in accept and "application/msgpack" not in accept


def format_response(response: BaseModel, status_code=200):
    """
    Helper function to format responses consistently based on client preference.

    Parameters
    ----------
    response : BaseModel
        The response object to format
    status_code : int
        HTTP status code (default: 200)

    Returns
    -------
    Response
        Formatted response in the client's preferred format
    """
    try:
        if wants_json(request):
            return JSONResponse(
                response.model_dump(mode="json"), status_code=status_code
            )

        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            status_code,
            {"Content-Type": "application/msgpack"},
        )
    except Exception as e:
        logger.error(f"Error formatting response: {e}", exc_info=True)
        # Fallback to JSON response if formatting fails
        return JSONResponse(
            {"error": "Response formatting failed", "details": str(e)}, status_code=500
        )
