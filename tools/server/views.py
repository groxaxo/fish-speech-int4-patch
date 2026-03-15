import os
import re
import tempfile
import time
from http import HTTPStatus
from pathlib import Path

import numpy as np
import ormsgpack
import torch
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    UploadFile,
    request,
)
from loguru import logger
from typing_extensions import Annotated

from fish_speech.text import normalize_text_for_tts
from fish_speech.utils.schema import (
    AddReferenceResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    OpenAISpeechRequest,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    UpdateReferenceResponse,
)
from tools.server.api_utils import (
    build_openai_error,
    build_openai_model_list,
    build_openai_tts_request,
    buffer_to_async_generator,
    chunk_bytes,
    format_response,
    get_openai_model,
    get_content_type,
    inference_async,
    prepare_tts_request,
    resolve_openai_reference_id,
    serialize_audio_output,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)
routes = Routes()


def _get_tts_context():
    app_state = request.app.state
    model_manager: ModelManager = app_state.model_manager
    engine = model_manager.tts_inference_engine
    sample_rate = engine.decoder_model.sample_rate
    return app_state, engine, sample_rate


def _audio_headers(audio_format: str, chunked: bool = False) -> dict[str, str]:
    headers = {
        "Content-Disposition": f"attachment; filename=audio.{audio_format}",
        "Cache-Control": "no-cache",
    }
    if chunked:
        headers["Transfer-Encoding"] = "chunked"
        headers["X-Accel-Buffering"] = "no"
    return headers


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    """
    Encode audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Encode the audio
        start_time = time.time()
        tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
        logger.info(
            f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms"
        )

        # Return the response
        return ormsgpack.packb(
            ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN encode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to encode audio"
        )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    """
    Decode tokens to audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Decode the audio
        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = batch_vqgan_decode(decoder_model, tokens)
        logger.info(
            f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms"
        )
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        # Return the response
        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN decode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to decode tokens to audio"
        )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    """
    Generate speech from text using TTS model.
    """
    try:
        app_state, engine, sample_rate = _get_tts_context()
        req = prepare_tts_request(req, app_state.max_text_length)

        if req.streaming:
            return StreamResponse(
                iterable=inference_async(req, engine),
                headers=_audio_headers(req.format, chunked=True),
                content_type=get_content_type(req.format),
            )

        fake_audios = next(inference(req, engine))
        audio_bytes = serialize_audio_output(fake_audios, sample_rate, req.format)
        return StreamResponse(
            iterable=buffer_to_async_generator(audio_bytes),
            headers=_audio_headers(req.format),
            content_type=get_content_type(req.format),
        )
    except ValueError as e:
        raise HTTPException(HTTPStatus.BAD_REQUEST, content=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


@routes.http.get("/v1/models")
async def list_models():
    return JSONResponse(build_openai_model_list())


@routes.http.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    model = get_openai_model(model_id)
    if model is None:
        raise HTTPException(
            HTTPStatus.NOT_FOUND,
            content=build_openai_error(
                "model_not_found",
                f"Model '{model_id}' not found",
                "invalid_request_error",
            ),
        )
    return JSONResponse(model)


@routes.http.get("/v1/audio/voices")
async def list_openai_voices():
    _, engine, _ = _get_tts_context()
    return JSONResponse({"voices": engine.list_reference_ids()})


@routes.http.post("/v1/audio/speech")
async def openai_speech(req: Annotated[OpenAISpeechRequest, Body(exclusive=True)]):
    try:
        if get_openai_model(req.model) is None:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=build_openai_error(
                    "invalid_model",
                    f"Unsupported model: {req.model}",
                    "invalid_request_error",
                ),
            )

        app_state, engine, sample_rate = _get_tts_context()
        available_reference_ids = engine.list_reference_ids()
        reference_id = resolve_openai_reference_id(req.voice, available_reference_ids)
        tts_req = build_openai_tts_request(req, reference_id)
        tts_req = prepare_tts_request(tts_req, app_state.max_text_length)

        if req.stream and tts_req.streaming:
            return StreamResponse(
                iterable=inference_async(tts_req, engine),
                headers=_audio_headers(tts_req.format, chunked=True),
                content_type=get_content_type(tts_req.format),
            )

        fake_audios = next(inference(tts_req, engine))
        audio_bytes = serialize_audio_output(fake_audios, sample_rate, tts_req.format)

        if req.stream:
            return StreamResponse(
                iterable=chunk_bytes(audio_bytes),
                headers=_audio_headers(tts_req.format, chunked=True),
                content_type=get_content_type(tts_req.format),
            )

        return StreamResponse(
            iterable=buffer_to_async_generator(audio_bytes),
            headers=_audio_headers(tts_req.format),
            content_type=get_content_type(tts_req.format),
        )
    except ValueError as e:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content=build_openai_error(
                "validation_error", str(e), "invalid_request_error"
            ),
        )
    except RuntimeError as e:
        logger.error(f"Processing error in /v1/audio/speech: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content=build_openai_error("processing_error", str(e), "server_error"),
        )
    except HTTPException as e:
        if isinstance(e.content, dict):
            raise
        raise HTTPException(
            e.status_code,
            content=build_openai_error(
                "validation_error" if e.status_code < 500 else "processing_error",
                str(e.content),
                "invalid_request_error" if e.status_code < 500 else "server_error",
            ),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /v1/audio/speech: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content=build_openai_error("processing_error", str(e), "server_error"),
        )


@routes.http.post("/v1/references/add")
async def add_reference(
    id: str = Body(...), audio: UploadFile = Body(...), text: str = Body(...)
):
    """
    Add a new reference voice with audio file and text.
    """
    temp_file_path = None

    try:
        # Validate input parameters
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")

        if not text or not text.strip():
            raise ValueError("Reference text cannot be empty")

        text = normalize_text_for_tts(text)
        if not text:
            raise ValueError("Reference text is empty after preprocessing")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Read the uploaded audio file
        audio_content = audio.read()
        if not audio_content:
            raise ValueError("Audio file is empty or could not be read")

        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        # Add the reference using the engine's reference loader
        engine.add_reference(id, temp_file_path, text)

        response = AddReferenceResponse(
            success=True,
            message=f"Reference voice '{id}' added successfully",
            reference_id=id,
        )
        return format_response(response)

    except FileExistsError as e:
        logger.warning(f"Reference ID '{id}' already exists: {e}")
        response = AddReferenceResponse(
            success=False,
            message=f"Reference ID '{id}' already exists",
            reference_id=id,
        )
        return format_response(response, status_code=409)  # Conflict

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{id}': {e}")
        response = AddReferenceResponse(success=False, message=str(e), reference_id=id)
        return format_response(response, status_code=400)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"File system error for reference '{id}': {e}")
        response = AddReferenceResponse(
            success=False, message="File system error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error adding reference '{id}': {e}", exc_info=True)
        response = AddReferenceResponse(
            success=False, message="Internal server error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(
                    f"Failed to clean up temporary file {temp_file_path}: {e}"
                )


@routes.http.get("/v1/references/list")
async def list_references():
    """
    Get a list of all available reference voice IDs.
    """
    try:
        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Get the list of reference IDs
        reference_ids = engine.list_reference_ids()

        response = ListReferencesResponse(
            success=True,
            reference_ids=reference_ids,
            message=f"Found {len(reference_ids)} reference voices",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Unexpected error listing references: {e}", exc_info=True)
        response = ListReferencesResponse(
            success=False, reference_ids=[], message="Internal server error occurred"
        )
        return format_response(response, status_code=500)


@routes.http.delete("/v1/references/delete")
async def delete_reference(reference_id: str = Body(...)):
    """
    Delete a reference voice by ID.
    """
    try:
        # Validate input parameters
        if not reference_id or not reference_id.strip():
            raise ValueError("Reference ID cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Delete the reference using the engine's reference loader
        engine.delete_reference(reference_id)

        response = DeleteReferenceResponse(
            success=True,
            message=f"Reference voice '{reference_id}' deleted successfully",
            reference_id=reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(f"Reference ID '{reference_id}' not found: {e}")
        response = DeleteReferenceResponse(
            success=False,
            message=f"Reference ID '{reference_id}' not found",
            reference_id=reference_id,
        )
        return format_response(response, status_code=404)  # Not Found

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False, message=str(e), reference_id=reference_id
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error deleting reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False,
            message="File system error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(
            f"Unexpected error deleting reference '{reference_id}': {e}", exc_info=True
        )
        response = DeleteReferenceResponse(
            success=False,
            message="Internal server error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/references/update")
async def update_reference(
    old_reference_id: str = Body(...), new_reference_id: str = Body(...)
):
    """
    Rename a reference voice directory from old_reference_id to new_reference_id.
    """
    try:
        # Validate input parameters
        if not old_reference_id or not old_reference_id.strip():
            raise ValueError("Old reference ID cannot be empty")
        if not new_reference_id or not new_reference_id.strip():
            raise ValueError("New reference ID cannot be empty")
        if old_reference_id == new_reference_id:
            raise ValueError("New reference ID must be different from old reference ID")

        # Validate ID format per ReferenceLoader rules
        id_pattern = r"^[a-zA-Z0-9\-_ ]+$"
        if not re.match(id_pattern, new_reference_id) or len(new_reference_id) > 255:
            raise ValueError(
                "New reference ID contains invalid characters or is too long"
            )

        # Access engine to update caches after renaming
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        refs_base = Path("references")
        old_dir = refs_base / old_reference_id
        new_dir = refs_base / new_reference_id

        # Existence checks
        if not old_dir.exists() or not old_dir.is_dir():
            raise FileNotFoundError(f"Reference ID '{old_reference_id}' not found")
        if new_dir.exists():
            # Conflict: destination already exists
            response = UpdateReferenceResponse(
                success=False,
                message=f"Reference ID '{new_reference_id}' already exists",
                old_reference_id=old_reference_id,
                new_reference_id=new_reference_id,
            )
            return format_response(response, status_code=409)

        # Perform rename
        old_dir.rename(new_dir)

        # Update in-memory cache key if present
        if old_reference_id in engine.ref_by_id:
            engine.ref_by_id[new_reference_id] = engine.ref_by_id.pop(old_reference_id)

        response = UpdateReferenceResponse(
            success=True,
            message=(
                f"Reference voice renamed from '{old_reference_id}' to '{new_reference_id}' successfully"
            ),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(str(e))
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as e:
        logger.warning(f"Invalid input for update reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error renaming reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message="File system error occurred",
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error updating reference: {e}", exc_info=True)
        response = UpdateReferenceResponse(
            success=False,
            message="Internal server error occurred",
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=500)
