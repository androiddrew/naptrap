"""PhonMatchNet network for single-clip inference.

Architecture after PhonMatchNet (Lee et al., Interspeech 2023) and NCSOFT's
BSD-3-Clause PyTorch implementation; see licenses/PhonMatchNet-BSD-3-Clause.txt.
Tensors hold one unpadded clip, so time axes are the valid lengths.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..features import KeywordMel


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        # Nested name kept so weight keys match the trained checkpoint.
        self.bn = nn.BatchNorm1d(128)


class _Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = KeywordMel()
        self.conv1 = nn.Conv1d(40, 128, 3, stride=2, padding=1)
        self.bn1 = _Norm()
        self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn2 = _Norm()


class PhonMatchNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = _Audio()
        self.google_upsample = nn.ConvTranspose1d(96, 128, 5, stride=4)
        self.text_projection = nn.Linear(256, 128)
        self.attention = nn.MultiheadAttention(128, 1, batch_first=True)
        self.utterance_gru = nn.GRU(128, 128, batch_first=True)
        self.utterance_head = nn.Linear(128, 1)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, keyword_mel, google, text):
        value = F.relu(self.audio.bn1.bn(self.audio.conv1(keyword_mel.transpose(1, 2))))
        value = F.relu(self.audio.bn2.bn(self.audio.conv2(value))).transpose(1, 2)
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
