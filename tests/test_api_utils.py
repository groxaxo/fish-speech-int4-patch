import unittest
import asyncio

import numpy as np

from fish_speech.text import TextNormalizationOptions
from fish_speech.utils.schema import OpenAISpeechRequest, ServeReferenceAudio, ServeTTSRequest
from tools.server.api_utils import (
    build_openai_model_list,
    build_openai_tts_request,
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


if __name__ == "__main__":
    unittest.main()
