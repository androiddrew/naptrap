"""Input validation and the 40-bin log-mel frontend used by PhonMatchNet."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

SAMPLE_RATE = 16000
MIN_SAMPLES = SAMPLE_RATE // 2
MAX_SAMPLES = SAMPLE_RATE * 2


def feature_tensor(value: Any, name: str, width: int, minimum: int, maximum: int) -> torch.Tensor:
    """Accept ``[time, width]`` or ``[1, time, width]`` float32 and return ``[1, time, width]`` on CPU."""
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be a float32 feature array") from error
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
    """Validate one mono 16 kHz clip of 0.5-2 s, float32 scaled into [-1, 1]."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Audio must be mono 16 kHz; resample before inference")
    try:
        wave = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("waveform must be a float32 array") from error
    if wave.dtype != torch.float32 or wave.ndim != 1:
        raise ValueError("waveform must be a one-dimensional float32 array scaled into [-1, 1]")
    if not MIN_SAMPLES <= wave.numel() <= MAX_SAMPLES:
        raise ValueError("Clip duration must be 0.5-2 seconds (8,000-32,000 samples at 16 kHz)")
    wave = wave.detach().to(device="cpu").contiguous()
    if not torch.isfinite(wave).all() or wave.abs().max() > 1:
        raise ValueError("waveform must contain finite samples in [-1, 1]")
    return wave


class KeywordMel(nn.Module):
    """Log10 power mel spectrogram: 400-sample window, 160 hop, 512 FFT, 40 bins.

    The mel matrix and window are stored with the trained weights.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mel_matrix", torch.zeros(257, 40))
        self.register_buffer("window", torch.zeros(400))

    @torch.inference_mode()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(wave, n_fft=512, hop_length=160, win_length=400,
                              window=self.window, center=True, pad_mode="reflect", return_complex=True)
        features = torch.log10(spectrum.abs().square().transpose(0, 1) @ self.mel_matrix + 1e-7)
        return features[:(wave.numel() + 159) // 160].unsqueeze(0)
