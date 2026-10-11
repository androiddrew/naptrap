"""Our selected PhonMatchNet, loaded without the original training repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ..bundle import BundleError, ModelBundle
from ..features import feature_tensor, waveform_tensor
from ..types import Prediction
from .network import PhonMatchNetwork


class PhonMatchNet:
    """CPU eager PyTorch inference with original seed-matched thresholds.

    Inputs are the exact Google speech_embedding/1 and neural G2P features used
    during training. This first backend performs content-only C-KWS detection.
    """

    name = "phonmatchnet"
    task = "C-KWS"

    @classmethod
    def from_bundle(cls, bundle: str | Path, *, seed: int = 17, google_encoder=None) -> "PhonMatchNet":
        return cls(bundle, seed=seed, google_encoder=google_encoder)

    def __init__(self, bundle: str | Path, *, seed: int = 17, google_encoder=None):
        self.bundle = ModelBundle(bundle)
        if seed not in (17, 29, 43) or seed not in self.bundle.manifest.get("seeds", []):
            raise BundleError("Choose a registered model seed: 17, 29 or 43")
        self.seed = int(seed)
        self.google_encoder = google_encoder
        directory = f"phonmatchnet/seed-{self.seed}"
        model_path = self.bundle.checked_file(directory + "/keyword.pt")
        frontend_path = self.bundle.checked_file(directory + "/keyword-frontend.pt")
        calibration = self.bundle.checked_json(directory + "/thresholds.json")
        identity = calibration.get("identity", {})
        if calibration.get("fit_split") != "dev-cal" or identity.get("seed") != self.seed or identity.get("variant") != self.name:
            raise BundleError("Calibration does not belong to this seed/model and dev-cal role")
        try:
            values = calibration["thresholds"]["keyword-only-no-gating"][self.task]
            self._thresholds = {far: float(values[str(far)]) for far in (.01, .1)}
        except (KeyError, TypeError, ValueError) as error:
            raise BundleError("Missing keyword-only C-KWS calibration") from error
        if not all(torch.isfinite(torch.tensor(v)) for v in self._thresholds.values()):
            raise BundleError("Nonfinite calibration threshold")
        frontend = torch.load(frontend_path, map_location="cpu", weights_only=True)
        if frontend.get("format") != "keyword-logmel-frontend-v1":
            raise BundleError("Unsupported keyword frontend format")
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        if payload.get("format") != "zp-kws-final-keyword-inference-v1" or payload.get("family") != self.name:
            raise BundleError("Model payload does not contain the registered PhonMatchNet state")
        self._model = PhonMatchNetwork(frontend["model"])
        self._model.load_state_dict(payload["model"], strict=True)
        for key, value in frontend["model"].items():
            if not torch.equal(self._model.audio.frontend.state_dict()[key], value):
                raise BundleError("Acoustic frontend differs between model and frontend payload")
        self._frontend = self._model.audio.frontend

    @property
    def thresholds(self) -> dict[float, float]:
        """A copy of the frozen operating points; no fitting happens here."""
        return dict(self._thresholds)

    @torch.inference_mode()
    def predict(self, features: Mapping[str, Any], *, far: float = .01) -> Prediction:
        """Score an unpadded clip from keyword_mel, google and text tensors."""
        if far not in self._thresholds:
            raise ValueError("far must select an existing development target: 0.01 or 0.1")
        missing = {"keyword_mel", "google", "text"} - features.keys()
        if missing:
            raise ValueError("Missing features: " + ", ".join(sorted(missing)))
        mel = feature_tensor(features["keyword_mel"], "keyword_mel", 40, 50, 200)
        google = feature_tensor(features["google"], "google", 96, 6, 25)
        text = feature_tensor(features["text"], "text", 256, 1, 128)
        if not 8 * google.shape[1] <= mel.shape[1] <= 8 * google.shape[1] + 8:
            raise ValueError("Google and keyword acoustic feature clocks disagree")
        logits = self._model(mel, google, text)
        if logits.shape != (1,) or not torch.isfinite(logits).all():
            raise RuntimeError("Model returned invalid keyword logits")
        score = float(logits.sigmoid()[0])
        threshold = self._thresholds[far]
        return Prediction(model=self.name, seed=self.seed, task=self.task, score=score,
                          threshold=threshold, accepted=score >= threshold, target_far=far,
                          details={"keyword_logit": float(logits[0]), "keyword_probability": score})

    def predict_audio(self, waveform: Any, *, text: Any, google: Any = None,
                      sample_rate: int = 16000, far: float = .01) -> Prediction:
        """Compute our mel branch from audio; callers supply matching neural features.

        ``google`` must describe this exact waveform/crop. ``text`` is the
        desired keyword's 256D neural G2P decoder sequence, not phoneme tokens.
        """
        wave = waveform_tensor(waveform, sample_rate)
        if google is None:
            if self.google_encoder is None:
                raise ValueError("Supply Google features or configure a GoogleSpeechEmbedding encoder")
            google = self.google_encoder(wave, sample_rate=sample_rate)
        features = feature_tensor(google, "google", 96, 6, 25)
        if features.shape[1] != wave.numel() // 1280:
            raise ValueError("Google features do not match the supplied waveform length")
        return self.predict({"keyword_mel": self._frontend(wave), "google": features, "text": text}, far=far)
