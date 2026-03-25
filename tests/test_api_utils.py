import unittest
import asyncio

import numpy as np

from fish_speech.text import TextNormalizationOptions
from fish_speech.utils.schema import OpenAISpeechRequest, ServeReferenceAudio, ServeTTSRequest
from tools.server.api_utils import (
    build_openai_error,
    build_openai_model_list,
    build_openai_tts_request,
    parse_args,
    prepare_tts_request,
    resolve_openai_reference_id,
    serialize_audio_output,
)


class ApiUtilsTests(unittest.TestCase):
    def test_prepare_tts_request_normalizes_text_and_references(self):
        req = ServeTTSRequest(
            text="Contact user@example.com",
            format="wav",
            references=[
                ServeReferenceAudio(audio=b"audio", text="Visit https://fish.audio/docs")
            ],
            normalization_options=TextNormalizationOptions(),
        )

        prepared = prepare_tts_request(req)

        self.assertIn("user at example dot com", prepared.text)
        self.assertIn("fish dot audio slash docs", prepared.references[0].text)

    def test_prepare_tts_request_rejects_empty_preprocessed_text(self):
        req = ServeTTSRequest(text="😀😀", format="wav")
        with self.assertRaisesRegex(ValueError, "Text is empty after preprocessing"):
            prepare_tts_request(req)

    def test_build_openai_tts_request_uses_buffered_streaming_for_mp3(self):
        req = OpenAISpeechRequest(
            input="Hello world",
            response_format="mp3",
            stream=True,
            normalization_options=TextNormalizationOptions(),
        )

        tts_req = build_openai_tts_request(req, reference_id=None)
        self.assertFalse(tts_req.streaming)
        self.assertEqual(tts_req.format, "mp3")

    def test_openai_speech_request_defaults_to_non_streaming(self):
        req = OpenAISpeechRequest(input="Hello world")
        self.assertFalse(req.stream)

    def test_prepare_tts_request_forces_spanish_language_hint(self):
        req = ServeTTSRequest(text="Hola señor, ¿cómo estás?", format="wav", language="en")
        prepared = prepare_tts_request(req)
        self.assertEqual(prepared.language, "es")

    def test_prepare_tts_request_keeps_explicit_english_language(self):
        req = ServeTTSRequest(text="Hello world", format="wav", language="en")
        prepared = prepare_tts_request(req)
        self.assertEqual(prepared.language, "en")

    def test_chunk_bytes_returns_async_iterable(self):
        from tools.server.api_utils import chunk_bytes

        async def collect():
            items = []
            async for chunk in chunk_bytes(b"abcdef", chunk_size=2):
                items.append(chunk)
            return items

        self.assertEqual(asyncio.run(collect()), [b"ab", b"cd", b"ef"])

    def test_openai_voice_alias_and_reference_lookup(self):
        self.assertIsNone(resolve_openai_reference_id("alloy", ["demo"]))
        self.assertEqual(resolve_openai_reference_id("demo", ["demo"]), "demo")
        with self.assertRaisesRegex(ValueError, "Voice 'missing' not found"):
            resolve_openai_reference_id("missing", ["demo"])

    def test_pcm_serialization_returns_int16_bytes(self):
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        data = serialize_audio_output(audio, sample_rate=24000, audio_format="pcm")
        self.assertEqual(len(data), 6)

    def test_openai_model_list_contains_openai_compatible_ids(self):
        payload = build_openai_model_list()
        model_ids = [item["id"] for item in payload["data"]]
        self.assertIn("tts-1", model_ids)
        self.assertIn("tts-1-hd", model_ids)
        self.assertIn("fish-speech", model_ids)
        self.assertIn("s2-pro", model_ids)

    def test_build_openai_error_matches_openai_error_shape(self):
        payload = build_openai_error("invalid_model", "Unsupported model")
        self.assertEqual(payload["error"]["code"], "invalid_model")
        self.assertEqual(payload["error"]["message"], "Unsupported model")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIsNone(payload["error"]["param"])

    def test_openai_speech_request_accepts_flac_output(self):
        req = OpenAISpeechRequest(input="Hello world", response_format="flac")
        tts_req = build_openai_tts_request(req, reference_id=None)
        self.assertEqual(tts_req.format, "flac")

    def test_api_server_defaults_to_bnb4_fp16_and_port_8880(self):
        args = parse_args([])
        self.assertTrue(args.bnb4)
        self.assertTrue(args.half)
        self.assertEqual(args.listen, "0.0.0.0:8880")


if __name__ == "__main__":
    unittest.main()
