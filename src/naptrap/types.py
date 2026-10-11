"""Common result and model interfaces for Naptrap backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class Prediction:
    """One short-clip decision at a frozen development operating point.

    ``score`` is model/task specific: PhonMatchNet currently returns keyword
    probability, while a future personalized model may return a fused score.
    ``target_far`` identifies calibration, not guaranteed deployment FAR.
    """

    model: str
    seed: int
    task: str
    score: float
    threshold: float
    accepted: bool
    target_far: float
    details: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "task": self.task,
            "score": self.score,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "target_far": self.target_far,
            "threshold_fit_split": "dev-cal",
            "details": dict(self.details),
        }


class KeywordModel(Protocol):
    """Small extension point; each backend documents its own feature inputs."""

    def predict(self, features: Mapping[str, Any], *, far: float = 0.01) -> Prediction:
        ...
