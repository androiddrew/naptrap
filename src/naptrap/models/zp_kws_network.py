"""ZP-KWS-style keyword matcher and EfficientTDNN speaker encoder for single-clip inference.

The method follows Hu et al., "Personalized Keyword Spotting for User-Defined
Keywords Leveraging Text-Independent Speaker Verification" (arXiv:2606.20106);
this is an independent implementation, not the authors' code. The keyword
matcher extends the PhonMatchNet design (NCSOFT, BSD-3-Clause; see
licenses/PhonMatchNet-BSD-3-Clause.txt). The speaker encoder is EfficientTDNN
Small from the Sugar toolkit (MIT; see licenses/Sugar-MIT.txt), written out as
the fixed sub-network the weights were trained in. Tensors hold one unpadded
clip, so time axes are the valid lengths.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..features import KeywordMel


class _Norm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Nested name kept so weight keys match the trained checkpoint.
        self.bn = nn.BatchNorm1d(channels)


class _Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = KeywordMel()
        self.conv1 = nn.Conv1d(40, 128, 5, stride=2, padding=2)
        self.bn1 = _Norm(128)
        self.conv2 = nn.Conv1d(128, 256, 5, padding=2)
        self.bn2 = _Norm(256)
        self.gru = nn.GRU(256, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(256, 128)


class KeywordMatcher(nn.Module):
    """``keyword_mel [1,T,40]``, ``google [1,G,96]``, ``text [1,P,256]`` -> one keyword logit."""

    def __init__(self):
        super().__init__()
        self.audio = _Audio()
        self.google_upsample = nn.ConvTranspose1d(96, 128, 5, stride=4)
        self.audio_norm = nn.LayerNorm(128)
        self.text_projection = nn.Linear(256, 128)
        self.attention = nn.MultiheadAttention(128, 1, batch_first=True)
        self.utterance_gru = nn.GRU(128, 128, batch_first=True)
        self.utterance_head = nn.Linear(128, 1)

    def forward(self, keyword_mel, google, text):
        value = F.relu(self.audio.bn1.bn(self.audio.conv1(keyword_mel.transpose(1, 2))))
        value = F.relu(self.audio.bn2.bn(self.audio.conv2(value))).transpose(1, 2)
        value, _ = self.audio.gru(value)
        value = self.audio_norm(self.audio.projection(value))
        pretrained = F.leaky_relu(self.google_upsample(google.transpose(1, 2))).transpose(1, 2)
        pretrained = F.pad(pretrained, (0, 0, 0, value.shape[1]))[:, :value.shape[1]]
        textual = F.leaky_relu(self.text_projection(text))
        joint = torch.cat([value + pretrained, textual], dim=1)
        q, k, v = F.linear(joint, self.attention.in_proj_weight, self.attention.in_proj_bias).chunk(3, dim=-1)
        logits = (q / math.sqrt(128.)) @ k.transpose(1, 2)
        mask = torch.triu(torch.ones((joint.shape[1], joint.shape[1]), dtype=torch.bool, device=joint.device), diagonal=1)
        attended = self.attention.out_proj(torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1) @ v)
        _, hidden = self.utterance_gru(attended)
        return self.utterance_head(hidden[-1]).squeeze(-1)


class SpeakerMel(nn.Module):
    """Instance-normalized natural-log mel power: pre-emphasis 0.97, Hamming 400/160, 512 FFT, 80 bins.

    The pre-emphasis kernel, window and mel matrix are stored with the trained weights.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("preemphasis", torch.zeros(1, 1, 2))
        self.register_buffer("window", torch.zeros(400))
        self.register_buffer("mel_matrix", torch.zeros(257, 80))

    @torch.inference_mode()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        emphasized = F.conv1d(F.pad(wave[None, None], (1, 0), mode="reflect"), self.preemphasis)[0]
        spectrum = torch.stft(emphasized, n_fft=512, hop_length=160, win_length=400, window=self.window,
                              center=True, pad_mode="reflect", return_complex=True)
        mel = spectrum.abs().square().transpose(1, 2) @ self.mel_matrix
        return F.instance_norm(torch.log(mel.transpose(1, 2) + 1e-6), eps=1e-5)


class _ConvReLUBN(nn.Module):
    def __init__(self, channels_in, channels_out, kernel=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(channels_in, channels_out, kernel, padding=dilation * (kernel - 1) // 2,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channels_out)

    def forward(self, value):
        return self.bn(F.relu(self.conv(value)))


class _Res2Conv(nn.Module):
    def __init__(self, channels, scale, dilation):
        super().__init__()
        self.width = channels // scale
        self.convs = nn.ModuleList(_ConvReLUBN(self.width, self.width, 3, dilation) for _ in range(scale - 1))

    def forward(self, value):
        parts = torch.split(value, self.width, dim=1)
        outputs, previous = [], None
        for part, conv in zip(parts, self.convs):
            previous = conv(part if previous is None else part + previous)
            outputs.append(previous)
        return torch.cat(outputs + [parts[-1]], dim=1)


class _SqueezeExcite(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.reduce = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

    def forward(self, value):
        weights = torch.sigmoid(self.expand(F.relu(self.reduce(value.mean(dim=2)))))
        return value * weights.unsqueeze(2)


class _Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.first_conv = _ConvReLUBN(channels, channels)
        self.resconv = _Res2Conv(channels, 8, dilation)
        self.last_conv = _ConvReLUBN(channels, channels)
        self.se = _SqueezeExcite(channels, 64)

    def forward(self, value):
        return self.se(self.last_conv(self.resconv(self.first_conv(value))))


class _AttentiveStatsPool(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.linear1 = nn.Conv1d(channels, bottleneck, 1, bias=False)
        self.linear2 = nn.Conv1d(bottleneck, channels, 1, bias=False)

    def forward(self, value):
        weights = torch.softmax(self.linear2(torch.tanh(self.linear1(value))), dim=2)
        mean = (weights * value).sum(dim=2)
        std = ((weights * value.square()).sum(dim=2) - mean.square()).clamp(min=1e-9).sqrt()
        return torch.cat([mean, std], dim=1)


class EfficientTDNNSmall(nn.Module):
    """``speaker_mel [1,80,S]`` -> raw 192-D speaker embedding ``[1,192]``."""

    def __init__(self):
        super().__init__()
        self.layer1 = _ConvReLUBN(80, 256, 3)
        self.blocks = nn.ModuleList([_Block(256, 2), _Block(256, 3)])
        self.catconv = nn.Conv1d(512, 400, 1, bias=False)
        self.pooling = _AttentiveStatsPool(400, 128)
        self.bnpool = nn.BatchNorm1d(800)
        self.linear = nn.Linear(800, 192)
        self.bnembed = nn.BatchNorm1d(192)

    def forward(self, speaker_mel):
        first = self.layer1(speaker_mel)
        second = self.blocks[0](first) + first
        third = self.blocks[1](second) + first + second
        value = F.relu(self.catconv(torch.cat([second, third], dim=1)))
        return self.bnembed(self.linear(self.bnpool(self.pooling(value))))


class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = SpeakerMel()
        self.backbone = EfficientTDNNSmall()


class ZpKwsNetwork(nn.Module):
    """Both trained networks, frozen in inference mode."""

    def __init__(self):
        super().__init__()
        self.keyword = KeywordMatcher()
        self.speaker = SpeakerEncoder()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)
