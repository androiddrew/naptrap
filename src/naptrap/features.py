"""Explicit PhonMatchNet feature contracts and its independent mel frontend."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def feature_tensor(value: Any, name: str, width: int, minimum: int, maximum: int) -> torch.Tensor:
    """Accept [time,width] or [1,time,width] FP32, then stage on CPU."""
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be an FP32 feature array") from error
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype float32")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != width:
        raise ValueError(f"{name} must have shape [time,{width}] or [1,time,{width}]")
    if not minimum <= tensor.shape[1] <= maximum:
        raise ValueError(f"{name} time length must be between {minimum} and {maximum}")
    tensor = tensor.detach().to(device="cpu").contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    return tensor


def waveform_tensor(value: Any, sample_rate: int) -> torch.Tensor:
    if sample_rate != 16000:
        raise ValueError("The current model requires mono 16 kHz audio; resample explicitly before inference")
    try:
        wave = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("waveform must be a float32 array") from error
    if wave.dtype != torch.float32 or wave.ndim != 1:
        raise ValueError("waveform must be a one-dimensional float32 array, scaled once into [-1,1]")
    if not 8000 <= wave.numel() <= 32000:
        raise ValueError("Query duration must be 0.5–2 seconds (8,000–32,000 samples)")
    wave = wave.detach().to(device="cpu").contiguous()
    if not torch.isfinite(wave).all() or wave.abs().max() > 1:
        raise ValueError("waveform must contain finite unit-scaled samples in [-1,1]")
    return wave


class KeywordMel(nn.Module):
    """The exact 40-bin crop frontend, using the trained model's mel/window buffers."""

    def __init__(self, state):
        super().__init__()
        for key, shape in (("mel_matrix", (257, 40)), ("window", (400,))):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape:
                raise ValueError(f"Invalid keyword frontend buffer: {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite keyword frontend buffer: {key}")
            self.register_buffer(key, value.detach().cpu().clone())

    @torch.inference_mode()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(wave, n_fft=512, hop_length=160, win_length=400,
                              window=self.window, center=True, pad_mode="reflect", return_complex=True)
        features = torch.log10(spectrum.abs().square().transpose(0, 1) @ self.mel_matrix + 1e-7)
        return features[:(wave.numel() + 159) // 160].unsqueeze(0)
