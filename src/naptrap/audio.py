"""Minimal WAV reading without NumPy or audio libraries."""

from __future__ import annotations

import array
import sys
import wave
from pathlib import Path

import torch

from .features import SAMPLE_RATE


def load_wav(path: str | Path) -> torch.Tensor:
    """Read a mono 16 kHz 16-bit PCM WAV file as float32 samples in [-1, 1)."""
    with wave.open(str(Path(path).expanduser()), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != SAMPLE_RATE:
            raise ValueError("Expected a mono, 16-bit PCM, 16 kHz WAV file; convert it first, "
                             "e.g. ffmpeg -i in.wav -ac 1 -ar 16000 -sample_fmt s16 out.wav")
        samples = array.array("h", stream.readframes(stream.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    return torch.tensor(samples, dtype=torch.float32) / 32768
