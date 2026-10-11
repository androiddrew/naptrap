import hashlib
import json

import pytest
import torch

from naptrap.bundle import BundleError, ModelBundle
from naptrap.features import feature_tensor, waveform_tensor
from naptrap.registry import available_models, load_model, register_model


def test_features_accept_both_single_clip_layouts_without_changing_values():
    original = torch.arange(6 * 96, dtype=torch.float32).reshape(6, 96)
    actual = feature_tensor(original, "google", 96, 6, 25)
    assert actual.shape == (1, 6, 96)
    torch.testing.assert_close(actual[0], original, rtol=0, atol=0)


@pytest.mark.parametrize("value", [
    torch.zeros(6, 96, dtype=torch.float64), torch.zeros(5, 96),
    torch.zeros(2, 6, 96), torch.zeros(6, 95), torch.full((6, 96), float("nan")),
])
def test_invalid_feature_contracts_are_rejected(value):
    with pytest.raises(ValueError):
        feature_tensor(value, "google", 96, 6, 25)


@pytest.mark.parametrize("value,rate", [
    (torch.zeros(7999), 16000), (torch.zeros(32001), 16000),
    (torch.zeros(16000, 2), 16000), (torch.ones(16000) * 2, 16000),
    (torch.zeros(16000, dtype=torch.int16), 16000), (torch.zeros(16000), 8000),
])
def test_waveform_contract_does_not_silently_rescale_trim_or_resample(value, rate):
    with pytest.raises(ValueError):
        waveform_tensor(value, rate)


def test_silence_is_a_valid_waveform():
    assert torch.equal(waveform_tensor(torch.zeros(8000), 16000), torch.zeros(8000))


def test_bundle_rejects_changed_file_and_escape(tmp_path):
    file = tmp_path / "artifact"
    file.write_bytes(b"known asset")
    manifest = {"release": "python-onnx-v1", "files": {"artifact": {
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    bundle = ModelBundle(tmp_path)
    assert bundle.checked_file("artifact") == file
    with pytest.raises(BundleError, match="leaves"):
        bundle.checked_file("../outside")
    file.write_bytes(b"changed asset")
    with pytest.raises(BundleError, match="checksum"):
        bundle.checked_file("artifact")


def test_bundle_requires_supported_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("[]")
    with pytest.raises(BundleError):
        ModelBundle(tmp_path)


def test_registry_preserves_existing_models_and_rejects_unknown():
    assert "phonmatchnet" in available_models()
    with pytest.raises(ValueError, match="already registered"):
        register_model("phonmatchnet", lambda **kwargs: object())
    with pytest.raises(ValueError, match="Unknown"):
        load_model("nonexistent")


def test_registry_accepts_an_explicit_new_factory(monkeypatch):
    import naptrap.registry as registry
    monkeypatch.setattr(registry, "_FACTORIES", registry._FACTORIES.copy())
    instance = object()
    register_model("future-model", lambda **kwargs: instance)
    assert load_model("future-model", bundle="example") is instance
