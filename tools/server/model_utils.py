import hashlib
import io
import re

import librosa
import torch
import torchaudio
from cachetools import LRUCache, cached

CACHE_MAXSIZE = 256
MICRO_BATCH_SIZE = 8
ASR_SAMPLE_RATE = 16000
HUGE_GAP_THRESHOLD = 4000


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.half)
def batch_encode(model, audios_list: list[bytes]):
    # Get sample rate from model
    if hasattr(model, "spec_transform"):
        sample_rate = model.spec_transform.sample_rate
    else:
        sample_rate = model.sample_rate

    audios: list[torch.Tensor] = [
        (
            torch.from_numpy(librosa.load(io.BytesIO(audio), sr=sample_rate)[0])[None]
            if isinstance(audio, bytes)
            else audio
        )
        for audio in audios_list
    ]

    lengths = torch.tensor([audio.shape[-1] for audio in audios], device=model.device)
    max_length = lengths.max().item()

    print(f"Encode max length: {max_length / sample_rate:.2f}s")

    padded = torch.stack(
        [
            torch.nn.functional.pad(audio, (0, int(max_length - audio.shape[-1])))
            for audio in audios
        ]
    ).to(model.device)

    features, feature_lengths = model.encode(padded, audio_lengths=lengths)
    features, feature_lengths = features.cpu(), feature_lengths.cpu()

    return [feature[..., :length] for feature, length in zip(features, feature_lengths)]


def _hash_audio_bytes(audio: bytes) -> str:
    return hashlib.blake2b(audio, digest_size=16).hexdigest()


@cached(
    cache=LRUCache(maxsize=CACHE_MAXSIZE),
    key=lambda model, audios: (
        str(model.device),
        tuple(_hash_audio_bytes(audio) for audio in audios),
    ),
)
def _cached_vqgan_batch_encode(model, audios: tuple[bytes, ...]):
    return batch_encode(model, list(audios))


def cached_vqgan_batch_encode(model, audios: list[bytes]):
    if not all(isinstance(audio, bytes) for audio in audios):
        return batch_encode(model, audios)
    return _cached_vqgan_batch_encode(model, tuple(audios))


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.half)
def batch_vqgan_decode(model, features):
    lengths = torch.tensor(
        [feature.shape[-1] for feature in features], device=model.device
    )
    max_length = lengths.max().item()
    padded = torch.stack(
        [
            torch.nn.functional.pad(feature, (0, max_length - feature.shape[-1]))
            for feature in features
        ]
    ).to(model.device)

    # If bs too large, we do micro batch decode
    audios, audio_lengths = [], []
    for i in range(0, padded.shape[0], MICRO_BATCH_SIZE):
        audio, audio_length = model.decode(
            padded[i : i + MICRO_BATCH_SIZE],
            feature_lengths=lengths[i : i + MICRO_BATCH_SIZE],
        )
        audios.append(audio)
        audio_lengths.append(audio_length)
    audios = torch.cat(audios, dim=0)
    audio_lengths = torch.cat(audio_lengths, dim=0)
    audios, audio_lengths = audios.cpu(), audio_lengths.cpu()

    return [audio[..., :length].numpy() for audio, length in zip(audios, audio_lengths)]
