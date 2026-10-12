import wave

import pytest
import torch

from naptrap import load_wav, normalize_keyword
from naptrap.features import feature_tensor, waveform_tensor


def test_features_accept_both_single_clip_layouts_without_changing_values():
    original = torch.arange(6 * 96, dtype=torch.float32).reshape(6, 96)
    actual = feature_tensor(original, "google", 96, 6, 25)
    assert actual.shape == (1, 6, 96)
    assert torch.equal(actual[0], original)


@pytest.mark.parametrize("value", [
    torch.zeros(6, 96, dtype=torch.float64), torch.zeros(5, 96),
    torch.zeros(2, 6, 96), torch.zeros(6, 95), torch.full((6, 96), float("nan")),
])
def test_invalid_features_are_rejected(value):
    with pytest.raises(ValueError):
        feature_tensor(value, "google", 96, 6, 25)


@pytest.mark.parametrize("value,rate", [
    (torch.zeros(7999), 16000), (torch.zeros(32001), 16000),
    (torch.zeros(16000, 2), 16000), (torch.ones(16000) * 2, 16000),
    (torch.zeros(16000, dtype=torch.int16), 16000), (torch.zeros(16000), 8000),
])
def test_waveforms_are_not_silently_rescaled_trimmed_or_resampled(value, rate):
    with pytest.raises(ValueError):
        waveform_tensor(value, rate)


def test_silence_is_a_valid_waveform():
    assert torch.equal(waveform_tensor(torch.zeros(8000), 16000), torch.zeros(8000))


@pytest.mark.parametrize("text,expected", [
    ("Hey, Jarvis!", "hey jarvis"),
    ("Café   Olé", "cafe ole"),
    ("don’t STOP", "don't stop"),
])
def test_keyword_normalization(text, expected):
    assert normalize_keyword(text) == expected


@pytest.mark.parametrize("text", ["", "!!!", "route 66"])
def test_unusable_keywords_are_rejected(text):
    with pytest.raises(ValueError):
        normalize_keyword(text)


def _write_wav(path, samples, channels=1, rate=16000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))


def test_load_wav_scales_16_bit_pcm(tmp_path):
    path = tmp_path / "clip.wav"
    _write_wav(path, [0, 16384, -32768, 32767])
    assert torch.equal(load_wav(path), torch.tensor([0, 0.5, -1, 32767 / 32768]))


@pytest.mark.parametrize("channels,rate", [(2, 16000), (1, 8000)])
def test_load_wav_rejects_other_formats(tmp_path, channels, rate):
    path = tmp_path / "clip.wav"
    _write_wav(path, [0] * 4, channels=channels, rate=rate)
    with pytest.raises(ValueError, match="16 kHz"):
        load_wav(path)
