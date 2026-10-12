"""Common result and model interfaces for naptrap backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol

import torch



@dataclass(frozen=True)
class Keyword:
    """A keyword's normalized text and its precomputed text features."""

    text: str
    features: torch.Tensor = field(repr=False, compare=False)


@dataclass(frozen=True)
class Prediction:
    """One clip's keyword decision at a fixed operating point.

    ``score`` is model specific; for PhonMatchNet it is the keyword-match
    probability. ``target_far`` names the calibration target the threshold was
    fitted for on development data. It is not a guaranteed false-accept rate on
    other audio.
    """

    model: str
    task: str
    score: float
    threshold: float
    accepted: bool
    target_far: float
    details: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "score": self.score,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "target_far": self.target_far,
            "details": dict(self.details),
        }


class KeywordModel(Protocol):
    """Interface every backend implements; inputs are documented per backend."""

    def keyword(self, text: str) -> Keyword:
        ...

    def detect(self, waveform: Any, keyword: "str | Keyword", *, sample_rate: int = 16000,
               far: float = 0.01) -> Prediction:
        ...

    def predict(self, features: Mapping[str, Any], *, far: float = 0.01) -> Prediction:
        ...
