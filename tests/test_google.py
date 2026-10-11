import os

import pytest
import torch

from naptrap import GoogleSpeechEmbedding, load_model


def test_random_unloaded_encoder_is_not_used_for_inference():
    model = GoogleSpeechEmbedding()
    with pytest.raises(RuntimeError, match="pretrained"):
        model(torch.zeros(8000))


@pytest.fixture(scope="module")
def encoder():
    directory = os.environ.get("NAPTRAP_TEST_GOOGLE")
    if not directory:
        pytest.skip("Set NAPTRAP_TEST_GOOGLE to test the converted pretrained encoder")
    return GoogleSpeechEmbedding.from_directory(directory)


@pytest.mark.parametrize("samples", [8000, 8001, 8960, 16000, 31999, 32000])
def test_google_clock_and_valid_silence(encoder, samples):
    values = encoder(torch.zeros(samples))
    assert values.shape == (samples // 1280, 96)
    assert torch.isfinite(values).all()
    assert not values.requires_grad


def test_encoder_stays_in_inference_mode(encoder):
    encoder.train(True)
    assert not encoder.training
    assert not any(m.training for m in encoder.norms)


def test_automatic_audio_encoder_matches_explicit_features(encoder):
    bundle = os.environ.get("NAPTRAP_TEST_BUNDLE")
    if not bundle:
        pytest.skip("Set NAPTRAP_TEST_BUNDLE for the complete model fixture")
    model = load_model(bundle=bundle, google_encoder=encoder)
    waveform = torch.zeros(8000)
    text = torch.zeros(6, 256)
    supplied = model.predict_audio(waveform, google=encoder(waveform), text=text)
    automatic = model.predict_audio(waveform, text=text)
    assert supplied == automatic
