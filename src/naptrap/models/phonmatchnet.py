"""PhonMatchNet text-defined keyword detection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch

from ..artifacts import WEIGHTS, ArtifactError, load_artifact, read_config, sha256
from ..features import SAMPLE_RATE, feature_tensor, waveform_tensor
from ..google import FORMAT as GOOGLE_FORMAT, GoogleSpeechEmbedding
from ..hub import DEFAULT_REVISION, download
from ..text import FORMAT as G2P_FORMAT, G2PTextEncoder, normalize_keyword
from ..types import Keyword, Prediction
from .network import PhonMatchNetwork

FORMAT = "naptrap-phonmatchnet-v1"
DEFAULT_REPO = "androiddrew/naptrap-phonmatchnet-en"
PathLike = Union[str, Path]


class PhonMatchNet:
    """Content-only (C-KWS) detection: was this keyword spoken in the clip?

    Inputs are one mono 16 kHz clip of 0.5-2 seconds and an English keyword.
    The keyword is converted to G2P decoder features, and the audio to 40-bin
    log-mels plus Google speech embeddings. The model has no speaker
    enrollment: it does not check who is speaking.
    """

    name = "phonmatchnet"
    task = "C-KWS"

    def __init__(self, model_dir: PathLike, google_dir: PathLike, g2p_dir: PathLike):
        config, state = load_artifact(model_dir, FORMAT)
        try:
            self._thresholds = {float(far): float(value) for far, value in config["thresholds"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ArtifactError("config.json has no valid thresholds") from error
        if not self._thresholds or not all(math.isfinite(v) for v in self._thresholds.values()):
            raise ArtifactError("config.json has no valid thresholds")
        self.config = config
        self._check_companion("google", google_dir, GOOGLE_FORMAT)
        self._check_companion("g2p", g2p_dir, G2P_FORMAT)
        self._model = PhonMatchNetwork()
        self._model.load_state_dict(state, strict=True)
        self._frontend = self._model.audio.frontend
        self.google_encoder = GoogleSpeechEmbedding.from_pretrained(google_dir)
        self.text_encoder = G2PTextEncoder.from_pretrained(g2p_dir)

    @classmethod
    def from_pretrained(cls, model_dir: Optional[PathLike] = None, *, google_dir: Optional[PathLike] = None,
                        g2p_dir: Optional[PathLike] = None, revision: Optional[str] = None,
                        cache_dir: Optional[str] = None) -> "PhonMatchNet":
        """Load from local directories, downloading any that are not given.

        Downloads need ``pip install 'naptrap[hub]'``. ``revision`` applies to
        the PhonMatchNet repository; companion encoders use the revisions
        pinned in its ``config.json``.
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
    def thresholds(self) -> Dict[float, float]:
        """Available operating points: development target FAR -> score threshold."""
        return dict(self._thresholds)

    def keyword(self, text: str) -> Keyword:
        """Precompute a keyword's text features; reuse the result across clips."""
        return Keyword(normalize_keyword(text), self.text_encoder(text))

    @torch.inference_mode()
    def detect(self, waveform: Any, keyword: Union[str, Keyword], *, sample_rate: int = SAMPLE_RATE,
               far: float = 0.01) -> Prediction:
        """Score one clip against a keyword string or a :meth:`keyword` result.

        ``waveform`` is a 1-D float32 tensor or array of 0.5-2 s mono 16 kHz
        audio scaled into [-1, 1]. ``far`` selects an operating point from
        :attr:`thresholds`.
        """
        if not isinstance(keyword, Keyword):
            keyword = self.keyword(keyword)
        wave = waveform_tensor(waveform, sample_rate)
        features = {
            "keyword_mel": self._frontend(wave),
            "google": self.google_encoder(wave, sample_rate=sample_rate),
            "text": keyword.features,
        }
        return self.predict(features, far=far)

    @torch.inference_mode()
    def predict(self, features: Mapping[str, Any], *, far: float = 0.01) -> Prediction:
        """Score precomputed ``keyword_mel`` [T,40], ``google`` [G,96] and ``text`` [P,256] features."""
        if far not in self._thresholds:
            choices = ", ".join(str(v) for v in sorted(self._thresholds))
            raise ValueError(f"far must be one of the calibrated operating points: {choices}")
        missing = {"keyword_mel", "google", "text"} - set(features.keys())
        if missing:
            raise ValueError("Missing features: " + ", ".join(sorted(missing)))
        mel = feature_tensor(features["keyword_mel"], "keyword_mel", 40, 50, 200)
        google = feature_tensor(features["google"], "google", 96, 6, 25)
        text = feature_tensor(features["text"], "text", 256, 1, 128)
        if not 8 * google.shape[1] <= mel.shape[1] <= 8 * google.shape[1] + 8:
            raise ValueError("keyword_mel and google features describe different clip lengths")
        logits = self._model(mel, google, text)
        if logits.shape != (1,) or not torch.isfinite(logits).all():
            raise RuntimeError("Model returned an invalid keyword logit")
        score = float(logits.sigmoid()[0])
        threshold = self._thresholds[far]
        return Prediction(model=self.name, task=self.task, score=score, threshold=threshold,
                          accepted=score >= threshold, target_far=far,
                          details={"keyword_logit": float(logits[0])})
