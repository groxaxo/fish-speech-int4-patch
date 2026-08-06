import os
from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


class VQManager:

    def __init__(self):
        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.load_audio: Callable

    def decode_vq_tokens(self, codes):
        logger.info(f"VQ features: {codes.shape}")

        if not isinstance(self.decoder_model, DAC):
            raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

        # Decoder activations scale with utterance length - the stack expands to
        # 44.1 kHz - so one long decode can need several times the memory the
        # weights do. Decoding in windows bounds that. The decoder and the
        # quantizer's post_module are causal, so a window only needs left
        # context; its output prefix is then discarded.
        max_frames = int(os.getenv("FISH_DECODE_WINDOW_FRAMES", "128"))
        context_frames = int(os.getenv("FISH_DECODE_CONTEXT_FRAMES", "32"))

        n_frames = codes.shape[-1]
        if max_frames <= 0 or n_frames <= max_frames:
            return self.decoder_model.from_indices(codes[None])[0].squeeze()

        logger.info(
            f"Decoding {n_frames} frames in windows of {max_frames} "
            f"(+{context_frames} context)"
        )
        segments = []
        for start in range(0, n_frames, max_frames):
            end = min(start + max_frames, n_frames)
            ctx_start = max(0, start - context_frames)
            window = codes[:, ctx_start:end]
            audio = self.decoder_model.from_indices(window[None])[0].squeeze()
            # Derive samples-per-frame from the decode itself rather than
            # assuming hop_length, then drop the context prefix.
            per_frame = audio.shape[-1] // window.shape[-1]
            segments.append(audio[(start - ctx_start) * per_frame :])

        return torch.cat(segments, dim=-1)

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Load audios, and prepare basic info here
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate
            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            audios = torch.from_numpy(reference_audio_content).to(
                self.decoder_model.device
            )[None, None, :]
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.decoder_model.device, dtype=torch.long
            )
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            # VQ Encoder
            if isinstance(self.decoder_model, DAC):
                prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][0]
                logger.info(f"Encoded prompt: {prompt_tokens.shape}")
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
