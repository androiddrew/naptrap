"""Streaming detection: feed audio as it arrives and get keyword detections back."""

from __future__ import annotations

import array
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .features import SAMPLE_RATE
from .types import Keyword, KeywordModel


@dataclass(frozen=True)
class Detection:
    """One keyword heard in a stream.

    ``start`` and ``end`` are the seconds, counted from the start of the stream,
    of the window whose score completed the detection. ``score`` is the highest
    score in the run of windows that led to it.
    """

    keyword: str
    score: float
    threshold: float
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return {"keyword": self.keyword, "score": self.score, "threshold": self.threshold,
                "start": self.start, "end": self.end}


class Listener:
    """Watch a continuous mono 16 kHz stream for one or more keywords.

    The models score single clips, so the listener keeps the last ``window``
    seconds of audio and scores them every ``hop`` seconds. Each score is the
    same number ``model.detect()`` returns for that window. A keyword is
    reported when ``patience`` windows in a row reach the threshold, and then
    not again for ``refractory`` seconds, so one utterance gives one detection.

    ``threshold`` defaults to the model's calibrated threshold for ``far``.
    That calibration is per clip: a listener scores several windows a second,
    so choose a threshold from recordings made in your own setting.

    >>> listener = Listener(model, "hey computer")
    >>> for chunk in microphone:                # float32 chunks of any length
    ...     for detection in listener.feed(chunk):
    ...         print(detection.keyword, detection.end)
    """

    def __init__(self, model: KeywordModel, keywords: Union[str, Keyword, Sequence[Union[str, Keyword]]], *,
                 threshold: Optional[float] = None, far: float = 0.01, window: float = 1.5, hop: float = 0.25,
                 patience: int = 1, refractory: float = 2.0):
        if isinstance(keywords, (str, Keyword)):
            keywords = [keywords]
        self._keywords = [k if isinstance(k, Keyword) else model.keyword(k) for k in keywords]
        if not self._keywords:
            raise ValueError("Give at least one keyword")
        if threshold is None:
            calibrated = getattr(model, "thresholds", {})
            if not isinstance(calibrated.get(far), float):
                raise ValueError("This model has no single threshold for that operating point; pass threshold=")
            threshold = calibrated[far]
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self._window, self._hop = round(window * SAMPLE_RATE), round(hop * SAMPLE_RATE)
        if self._window < 1 or self._hop < 1:
            raise ValueError("window and hop must be positive")
        if int(patience) != patience or patience < 1:
            raise ValueError("patience must be a whole number of windows, at least 1")
        if refractory < 0:
            raise ValueError("refractory must not be negative")
        self.model = model
        self.threshold = float(threshold)
        self.far, self.patience, self.refractory = far, int(patience), float(refractory)
        # Models that expose their audio features compute them once per window for all keywords.
        self._shared = callable(getattr(model, "audio_features", None))
        self.reset()

    def reset(self) -> None:
        """Forget the audio heard so far and start a new stream at time zero."""
        self._buffer = torch.zeros(0, dtype=torch.float32)
        self._samples = 0
        self._until_next = self._window  # the first score needs one full window
        self._runs = [0] * len(self._keywords)
        self._peaks = [0.0] * len(self._keywords)
        self._quiet_until = [0.0] * len(self._keywords)
        self._scores: Dict[str, float] = {}

    @property
    def keywords(self) -> List[str]:
        return [k.text for k in self._keywords]

    @property
    def time(self) -> float:
        """Seconds of audio fed since the stream started."""
        return self._samples / SAMPLE_RATE

    @property
    def scores(self) -> Dict[str, float]:
        """The most recent window's score for each keyword; empty until one window has been heard."""
        return dict(self._scores)

    def feed(self, samples: Any) -> List[Detection]:
        """Add the next samples of the stream: 1-D float32 in [-1, 1], any length.

        Returns the detections completed by this audio, usually none.
        """
        try:
            chunk = torch.as_tensor(samples)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("samples must be a float32 array") from error
        if chunk.dtype != torch.float32 or chunk.ndim != 1:
            raise ValueError("samples must be a one-dimensional float32 array scaled into [-1, 1]")
        chunk = chunk.detach().to(device="cpu")
        if chunk.numel() and (not torch.isfinite(chunk).all() or chunk.abs().max() > 1):
            raise ValueError("samples must be finite and in [-1, 1]")
        detections: List[Detection] = []
        position = 0
        while position < chunk.numel():
            take = min(self._until_next, chunk.numel() - position)
            self._buffer = torch.cat([self._buffer, chunk[position:position + take]])[-self._window:]
            self._samples += take
            self._until_next -= take
            position += take
            if self._until_next == 0:
                self._until_next = self._hop
                detections.extend(self._evaluate())
        return detections

    def feed_pcm16(self, data: bytes) -> List[Detection]:
        """Add raw 16-bit little-endian mono 16 kHz PCM, as most microphone libraries deliver it."""
        if len(data) % 2:
            raise ValueError("16-bit PCM must be a whole number of samples")
        values = array.array("h", data)
        if sys.byteorder == "big":
            values.byteswap()
        return self.feed(torch.tensor(values, dtype=torch.float32) / 32768)

    def _window_scores(self, clip: torch.Tensor) -> List[float]:
        if self._shared:
            audio = self.model.audio_features(clip)
            return [self.model.predict({**audio, "text": k.features}, far=self.far).score for k in self._keywords]
        return [self.model.detect(clip, k, far=self.far).score for k in self._keywords]

    def _evaluate(self) -> List[Detection]:
        end = self.time
        scores = self._window_scores(self._buffer.contiguous())
        self._scores = {k.text: s for k, s in zip(self._keywords, scores)}
        detections = []
        for index, (keyword, score) in enumerate(zip(self._keywords, scores)):
            if score < self.threshold:
                self._runs[index], self._peaks[index] = 0, 0.0
                continue
            self._runs[index] += 1
            self._peaks[index] = max(self._peaks[index], score)
            if self._runs[index] >= self.patience and end >= self._quiet_until[index]:
                detections.append(Detection(keyword.text, self._peaks[index], self.threshold,
                                            max(0.0, end - self._window / SAMPLE_RATE), end))
                self._quiet_until[index] = end + self.refractory
        return detections
