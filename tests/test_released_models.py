"""Checks against the released weights; set NAPTRAP_TEST_MODELS to run them."""

import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from naptrap import G2PTextEncoder, GoogleSpeechEmbedding, load_model, load_wav

# Reference outputs of the original training-time pipeline for the bundled example.
GOLDEN_SCORE = 0.9978187084197998
GOLDEN_THRESHOLD = 0.7149204015731812


@pytest.fixture(scope="module")
def model(real_models):
    return load_model("phonmatchnet", **real_models)


@pytest.fixture(scope="module")
def example(real_models):
    directory = real_models["model_dir"] / "examples"
    return load_wav(directory / "said-mister.wav"), load_file(str(directory / "said-mister.safetensors"))


def test_reference_features_reproduce_the_original_score(model, example):
    _, features = example
    result = model.predict(features)
    assert result.score == pytest.approx(GOLDEN_SCORE, abs=2e-6)
    assert result.threshold == GOLDEN_THRESHOLD and result.accepted


def test_audio_and_text_pipeline_matches_reference_features(model, example):
    waveform, features = example
    torch.testing.assert_close(model.keyword("said mister").features, features["text"], rtol=0, atol=1e-5)
    torch.testing.assert_close(model.google_encoder(waveform), features["google"], rtol=0, atol=1e-3)
    result = model.detect(waveform, "said mister")
    assert result.score == pytest.approx(GOLDEN_SCORE, abs=2e-5) and result.accepted
    assert not model.detect(waveform, "hey computer").accepted


@pytest.mark.parametrize("samples", [8000, 8001, 8960, 16000, 31999, 32000])
def test_google_clock_and_valid_silence(real_models, samples):
    encoder = GoogleSpeechEmbedding.from_pretrained(real_models["google_dir"])
    values = encoder(torch.zeros(samples))
    assert values.shape == (samples // 1280, 96) and torch.isfinite(values).all()


def test_encoders_stay_frozen(real_models):
    encoder = GoogleSpeechEmbedding.from_pretrained(real_models["google_dir"])
    encoder.train(True)
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    text = G2PTextEncoder.from_pretrained(real_models["g2p_dir"])
    assert not any(p.requires_grad for p in text.parameters())


def test_inference_needs_only_torch_and_safetensors(real_models, example):
    script = f"""
import importlib.abc, sys
BLOCKED = {{"numpy", "tensorflow", "torchaudio", "nltk", "scipy", "onnxruntime", "huggingface_hub"}}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("blocked: " + name)
sys.meta_path.insert(0, Block())
import naptrap
model = naptrap.load_model("phonmatchnet", model_dir={str(real_models['model_dir'])!r},
                           google_dir={str(real_models['google_dir'])!r}, g2p_dir={str(real_models['g2p_dir'])!r})
wave = naptrap.load_wav({str(real_models['model_dir'] / 'examples/said-mister.wav')!r})
assert model.detect(wave, "said mister").accepted
"""
    completed = subprocess.run([sys.executable, "-W", "ignore", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
