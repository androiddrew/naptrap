"""Keyword text features: a PyTorch port of the g2p_en neural decoder states.

PhonMatchNet consumes the hidden states of the g2p_en grapheme-to-phoneme
decoder (Kyubyong Park & Jongseok Kim, Apache-2.0), not phoneme strings. Each
word is encoded with the grapheme GRU, then decoded greedily; every emitted
decoder state becomes one 256-D feature row, and words are separated by the
decoder's padding embedding. This mirrors ``G2p.embedding()`` from the
PhonMatchNet repository without NumPy, NLTK or dictionary lookups.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import torch
from torch import nn

from .artifacts import load_artifact

FORMAT = "naptrap-g2p-en-decoder-v1"
GRAPHEMES = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
_START, _END, _MAX_STEPS = 2, 3, 20
_TOKENS = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+")


def normalize_keyword(text: str) -> str:
    """Lowercase, strip accents and punctuation; the policy used during training."""
    if not isinstance(text, str):
        raise TypeError("keyword must be a string")
    text = unicodedata.normalize("NFKD", text.replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    words = _TOKENS.findall(text)
    if any(word.isdigit() for word in words):
        raise ValueError("Spell out numbers in the keyword, e.g. 'route sixty six'")
    if not words:
        raise ValueError("keyword contains no English letters")
    return " ".join(words)


class _GRUCell(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.w_ih = nn.Parameter(torch.zeros(3 * size, size))
        self.w_hh = nn.Parameter(torch.zeros(3 * size, size))
        self.b_ih = nn.Parameter(torch.zeros(3 * size))
        self.b_hh = nn.Parameter(torch.zeros(3 * size))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Same gate order and arithmetic as g2p_en's NumPy grucell (r, z | n).
        ih = x @ self.w_ih.T + self.b_ih
        hh = h @ self.w_hh.T + self.b_hh
        split = ih.shape[-1] * 2 // 3
        r, z = torch.sigmoid(ih[:, :split] + hh[:, :split]).chunk(2, dim=-1)
        n = torch.tanh(ih[:, split:] + r * hh[:, split:])
        return (1 - z) * n + z * h


class G2PTextEncoder(nn.Module):
    """Keyword string -> ``[positions, 256]`` float32 decoder features."""

    def __init__(self, size: int = 256, phonemes: int = 74):
        super().__init__()
        self.size = size
        self.enc_emb = nn.Parameter(torch.zeros(len(GRAPHEMES), size))
        self.encoder = _GRUCell(size)
        self.dec_emb = nn.Parameter(torch.zeros(phonemes, size))
        self.decoder = _GRUCell(size)
        self.fc_w = nn.Parameter(torch.zeros(phonemes, size))
        self.fc_b = nn.Parameter(torch.zeros(phonemes))
        self._index = {g: i for i, g in enumerate(GRAPHEMES)}
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "G2PTextEncoder":
        config, state = load_artifact(directory, FORMAT)
        model = cls(size=int(config["hidden_size"]), phonemes=int(config["phonemes"]))
        model.load_state_dict(state, strict=True)
        return model

    def _word(self, word: str) -> torch.Tensor:
        unknown = self._index["<unk>"]
        ids = [self._index.get(c, unknown) for c in word] + [self._index["</s>"]]
        h = torch.zeros(1, self.size)
        for index in ids:
            h = self.encoder(self.enc_emb[index:index + 1], h)
        token, states = _START, []
        for _ in range(_MAX_STEPS):
            h = self.decoder(self.dec_emb[token:token + 1], h)
            token = int((h @ self.fc_w.T + self.fc_b).argmax())
            if token == _END:
                break
            states.append(h)
        return torch.cat(states) if states else h.new_zeros(0, self.size)

    @torch.inference_mode()
    def forward(self, keyword: str) -> torch.Tensor:
        rows = []
        for word in normalize_keyword(keyword).split():
            rows += [self._word(word), self.dec_emb[0:1]]
        features = torch.cat(rows[:-1])
        if features.shape[0] < 1:
            raise ValueError("keyword produced no phonetic features")
        return features.contiguous()
