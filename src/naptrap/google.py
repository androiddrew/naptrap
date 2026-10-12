"""PyTorch port of Google's speech_embedding/1 (Apache-2.0) feature extractor.

The converted weights, mel matrix and window come from the original TensorFlow
model unchanged; this module never imports TensorFlow or NumPy.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import ArtifactError, load_artifact
from .features import SAMPLE_RATE, waveform_tensor

FORMAT = "naptrap-google-speech-embedding-v1"
PARENT_SHA256 = "0f626c42009c1eb916401b7b2800229a211e43100115bc591b9afbf1b60da27f"


class GoogleSpeechEmbedding(nn.Module):
    """16 kHz clip -> ``[floor(samples / 1280), 96]`` speech embeddings.

    The clip is left-padded with 11,120 zeros and not trimmed, matching how the
    keyword models were trained.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.zeros(400))
        self.register_buffer("mel_matrix", torch.zeros(257, 32))
        self.initial = nn.Conv2d(1, 24, (3, 3), padding=(0, 1), bias=False)
        self.norms = nn.ModuleList([nn.BatchNorm2d(24, eps=.001)])
        self.convs = nn.ModuleList()
        channels = 24
        for index in range(18):
            target = {2: 48, 6: 72, 10: 96}.get(index, channels)
            frequency = index % 2 == 0
            self.convs.append(nn.Conv2d(channels, target, (1, 3) if frequency else (3, 1),
                                        padding=(0, 1) if frequency else 0, bias=False))
            self.norms.append(nn.BatchNorm2d(target, eps=.001))
            channels = target
        self.output = nn.Conv2d(96, 96, (3, 1), bias=False)
        self._loaded = False
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # A frozen pretrained feature extractor; BatchNorm stays in inference mode.
        return super().train(False)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "GoogleSpeechEmbedding":
        config, state = load_artifact(directory, FORMAT)
        if config.get("source", {}).get("archive_sha256") != PARENT_SHA256:
            raise ArtifactError("Google speech embedding was converted from an unexpected source archive")
        model = cls()
        model.load_state_dict(state, strict=True)
        model._loaded = True
        return model.eval()

    @staticmethod
    def activation(value):
        # Original graph: max(max(0.2*x,x),-0.4), after inference BatchNorm.
        return F.leaky_relu(value, negative_slope=.2).clamp(min=-.4)

    def log_mel(self, waveform):
        padded = F.pad(waveform, (11120, 0))
        # TensorFlow STFT windows start at frame zero: no centered FFT window.
        frames = padded.unfold(0, 400, 160) * self.window
        magnitude = torch.fft.rfft(frames, n=512, dim=-1).abs()
        return torch.log(magnitude @ self.mel_matrix + .001)

    def encode_log_mel(self, features):
        value = features.unsqueeze(0).unsqueeze(0)
        value = self.activation(self.norms[0](F.relu(self.initial(value))))
        for index, conv in enumerate(self.convs):
            value = self.activation(self.norms[index + 1](conv(value)))
            if index in (1, 9, 17):
                value = F.max_pool2d(value, kernel_size=(2, 2), stride=(2, 2))
            elif index in (5, 13):
                # The fixed 32-bin input gives even frequency widths here, so the
                # original SAME pooling at index 5 adds no padding.
                value = F.max_pool2d(value, kernel_size=(1, 2), stride=(1, 2))
        return self.output(value)[0, :, :, 0].transpose(0, 1)

    @torch.inference_mode()
    def forward(self, waveform, sample_rate=SAMPLE_RATE):
        if not self._loaded:
            raise RuntimeError("Load pretrained weights with GoogleSpeechEmbedding.from_pretrained()")
        wave = waveform_tensor(waveform, sample_rate).to(self.window.device)
        result = self.encode_log_mel(self.log_mel(wave))
        if result.shape != (wave.numel() // 1280, 96) or not torch.isfinite(result).all():
            raise RuntimeError("Google speech embedding produced an invalid output")
        return result
