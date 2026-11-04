"""Experimental ZP-KWS-style personalized keyword detection.

Enabled only when the ``NAPTRAP_EXPERIMENTAL`` environment variable lists
``zp-kws``; see :mod:`naptrap.registry`. Not part of a released naptrap yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch.nn import functional as F

from ..artifacts import WEIGHTS, ArtifactError, load_artifact, read_config, sha256
from ..features import SAMPLE_RATE, feature_tensor, waveform_tensor
from ..google import FORMAT as GOOGLE_FORMAT, GoogleSpeechEmbedding
from ..hub import DEFAULT_REVISION, download
from ..text import FORMAT as G2P_FORMAT, G2PTextEncoder, normalize_keyword
from ..types import Keyword, Prediction
from .zp_kws_network import ZpKwsNetwork

FORMAT = "naptrap-zp-kws-v1"
DEFAULT_REPO = "androiddrew/naptrap-zp-kws-en"
TASKS = ("TO-KWS", "C-KWS")
PathLike = Union[str, Path]


@dataclass(frozen=True)
class Enrollment:
    """An enrolled speaker: the speaker encoder's embedding of one clip."""

    embedding: torch.Tensor = field(repr=False, compare=False)


class ZpKws:
    """Personalized detection (TO-KWS): was this keyword spoken by the enrolled speaker?

    Inputs are one mono 16 kHz query clip of 0.5-2 seconds, an English keyword
    and an :class:`Enrollment` made from a 0.5-2 s clip of the user with
    :meth:`enroll`. Without an enrollment, :meth:`detect` falls back to
    content-only detection (C-KWS) with the keyword matcher alone.

    The score is ``sigmoid(keyword_logit) * sigmoid(10 * cosine - 5)``, where
    ``cosine`` compares the query's and the enrollment's speaker embeddings.
    """

    name = "zp-kws"
    task = "TO-KWS"

    def __init__(self, model_dir: PathLike, google_dir: PathLike, g2p_dir: PathLike):
        config, state = load_artifact(model_dir, FORMAT)
        try:
            self._thresholds = {task: {float(far): float(value) for far, value in config["thresholds"][task].items()}
                                for task in TASKS}
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ArtifactError("config.json has no valid thresholds") from error
        if not all(values and all(math.isfinite(v) for v in values.values()) for values in self._thresholds.values()):
            raise ArtifactError("config.json has no valid thresholds")
        self.config = config
        self._check_companion("google", google_dir, GOOGLE_FORMAT)
        self._check_companion("g2p", g2p_dir, G2P_FORMAT)
        self._model = ZpKwsNetwork()
        self._model.load_state_dict(state, strict=True)
        self.google_encoder = GoogleSpeechEmbedding.from_pretrained(google_dir)
        self.text_encoder = G2PTextEncoder.from_pretrained(g2p_dir)

    @classmethod
    def from_pretrained(cls, model_dir: Optional[PathLike] = None, *, google_dir: Optional[PathLike] = None,
                        g2p_dir: Optional[PathLike] = None, revision: Optional[str] = None,
                        cache_dir: Optional[str] = None) -> "ZpKws":
        """Load from local directories, downloading any that are not given.

        Downloads need ``pip install 'naptrap[hub]'``. ``revision`` applies to
        the ZP-KWS repository; companion encoders use the revisions pinned in
        its ``config.json``.
        """
        if model_dir is None:
            model_dir = download(DEFAULT_REPO, revision or DEFAULT_REVISION, cache_dir)
        _, config = read_config(model_dir, FORMAT)
        requires = config.get("requires", {})
        if google_dir is None:
            google_dir = download(requires["google"]["repo_id"], requires["google"]["revision"], cache_dir)
        if g2p_dir is None:
            g2p_dir = download(requires["g2p"]["repo_id"], requires["g2p"]["revision"], cache_dir)
        return cls(model_dir, google_dir, g2p_dir)

    def _check_companion(self, key: str, directory: PathLike, expected_format: str) -> None:
        required = self.config.get("requires", {}).get(key, {})
        root, _ = read_config(directory, expected_format)
        if required.get("sha256") and sha256(root / WEIGHTS) != required["sha256"]:
            raise ArtifactError(f"{root} is not the {key} encoder this model was trained with")

    @property
    def thresholds(self) -> Dict[str, Dict[float, float]]:
        """Available operating points per task: development target FAR -> score threshold."""
        return {task: dict(values) for task, values in self._thresholds.items()}

    def keyword(self, text: str) -> Keyword:
        """Precompute a keyword's text features; reuse the result across clips."""
        return Keyword(normalize_keyword(text), self.text_encoder(text))

    @torch.inference_mode()
    def enroll(self, waveform: Any, *, sample_rate: int = SAMPLE_RATE) -> Enrollment:
        """Enroll a speaker from one 0.5-2 s mono 16 kHz clip; reuse the result across queries."""
        return self.enroll_features(self._model.speaker.frontend(waveform_tensor(waveform, sample_rate)))

    @torch.inference_mode()
    def enroll_features(self, speaker_mel: Any) -> Enrollment:
        """Enroll a speaker from precomputed ``speaker_mel`` [80,S] features."""
        return Enrollment(self._embed(speaker_mel))

    @torch.inference_mode()
    def detect(self, waveform: Any, keyword: Union[str, Keyword], *, enrollment: Optional[Enrollment] = None,
               sample_rate: int = SAMPLE_RATE, far: float = 0.01) -> Prediction:
        """Score one clip against a keyword and, if given, an enrolled speaker.

        ``waveform`` is a 1-D float32 tensor or array of 0.5-2 s mono 16 kHz
        audio scaled into [-1, 1]. ``far`` selects an operating point from
        :attr:`thresholds` for the task being scored.
        """
        if not isinstance(keyword, Keyword):
            keyword = self.keyword(keyword)
        wave = waveform_tensor(waveform, sample_rate)
        features = {
            "keyword_mel": self._model.keyword.audio.frontend(wave),
            "google": self.google_encoder(wave, sample_rate=sample_rate),
            "text": keyword.features,
        }
        if enrollment is not None:
            features["speaker_mel"] = self._model.speaker.frontend(wave)
        return self.predict(features, enrollment=enrollment, far=far)

    @torch.inference_mode()
    def predict(self, features: Mapping[str, Any], *, enrollment: Optional[Enrollment] = None,
                far: float = 0.01) -> Prediction:
        """Score precomputed ``keyword_mel`` [T,40], ``google`` [G,96] and ``text`` [P,256] features.

        With an ``enrollment``, ``speaker_mel`` [80,S] of the same clip is also required.
        """
        task = "C-KWS" if enrollment is None else "TO-KWS"
        if far not in self._thresholds[task]:
            choices = ", ".join(str(v) for v in sorted(self._thresholds[task]))
            raise ValueError(f"far must be one of the calibrated operating points: {choices}")
        required = {"keyword_mel", "google", "text"} | ({"speaker_mel"} if enrollment is not None else set())
        missing = required - set(features.keys())
        if missing:
            raise ValueError("Missing features: " + ", ".join(sorted(missing)))
        if enrollment is not None and not isinstance(enrollment, Enrollment):
            raise TypeError("enrollment must come from ZpKws.enroll()")
        mel = feature_tensor(features["keyword_mel"], "keyword_mel", 40, 50, 200)
        google = feature_tensor(features["google"], "google", 96, 6, 25)
        text = feature_tensor(features["text"], "text", 256, 1, 128)
        if not 8 * google.shape[1] <= mel.shape[1] <= 8 * google.shape[1] + 8:
            raise ValueError("keyword_mel and google features describe different clip lengths")
        logits = self._model.keyword(mel, google, text)
        if logits.shape != (1,) or not torch.isfinite(logits).all():
            raise RuntimeError("Model returned an invalid keyword logit")
        keyword_probability = float(logits.sigmoid()[0])
        details = {"keyword_logit": float(logits[0]), "keyword_probability": keyword_probability}
        score = keyword_probability
        if enrollment is not None:
            speaker_mel = _speaker_mel_tensor(features["speaker_mel"])
            if not mel.shape[1] <= speaker_mel.shape[2] <= mel.shape[1] + 1:
                raise ValueError("keyword_mel and speaker_mel features describe different clip lengths")
            query = self._embed(speaker_mel)
            cosine = float(F.cosine_similarity(query, enrollment.embedding, dim=0))
            speaker_probability = 1 / (1 + math.exp(-(10 * cosine - 5)))
            details.update(speaker_cosine=cosine, speaker_probability=speaker_probability)
            score = keyword_probability * speaker_probability
        threshold = self._thresholds[task][far]
        return Prediction(model=self.name, task=task, score=score, threshold=threshold,
                          accepted=score >= threshold, target_far=far, details=details)

    def _embed(self, speaker_mel: Any) -> torch.Tensor:
        mel = _speaker_mel_tensor(speaker_mel)
        embedding = self._model.speaker.backbone(mel)
        if embedding.shape != (1, 192) or not torch.isfinite(embedding).all():
            raise RuntimeError("Model returned an invalid speaker embedding")
        return embedding[0]


def _speaker_mel_tensor(value: Any) -> torch.Tensor:
    """Accept ``[80, S]`` or ``[1, 80, S]`` float32 (S = 51-201, a 0.5-2 s clip) and return ``[1, 80, S]``."""
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("speaker_mel must be a float32 feature array") from error
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dtype != torch.float32 or tensor.ndim != 3 or tuple(tensor.shape[:2]) != (1, 80):
        raise ValueError("speaker_mel must be float32 with shape [80,frames] or [1,80,frames]")
    if not 51 <= tensor.shape[2] <= 201:
        raise ValueError("speaker_mel frame count must be between 51 and 201")
    tensor = tensor.detach().to(device="cpu").contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError("speaker_mel must contain finite values")
    return tensor
