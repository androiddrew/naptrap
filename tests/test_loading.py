import json
import shutil
import sys

import pytest
import torch

from naptrap import ArtifactError, Keyword, PhonMatchNet, available_models, load_model, register_model
from naptrap.cli import main


def test_detect_runs_end_to_end_on_random_weights(tiny_models):
    model = load_model("phonmatchnet", **tiny_models)
    keyword = model.keyword("Hey Computer")
    assert isinstance(keyword, Keyword) and keyword.text == "hey computer"
    assert keyword.features.shape[1] == 256
    result = model.detect(torch.zeros(16000), keyword)
    assert 0 <= result.score <= 1 and result.threshold == 0.9 and result.task == "C-KWS"
    assert result == model.detect(torch.zeros(16000), "hey computer")
    assert set(result.to_dict()) == {"model", "task", "score", "threshold", "accepted", "target_far", "details"}


def test_thresholds_are_a_copy_and_unknown_operating_points_fail(tiny_models):
    model = PhonMatchNet.from_pretrained(**tiny_models)
    thresholds = model.thresholds
    thresholds[0.01] = 0
    assert model.thresholds == {0.01: 0.9, 0.1: 0.5}
    with pytest.raises(ValueError, match="far"):
        model.detect(torch.zeros(16000), "stop", far=0.05)


def test_mismatched_feature_clocks_fail(tiny_models):
    model = PhonMatchNet.from_pretrained(**tiny_models)
    with pytest.raises(ValueError, match="clip lengths"):
        model.predict({"keyword_mel": torch.zeros(100, 40), "google": torch.zeros(25, 96), "text": torch.zeros(3, 256)})


def test_changed_weights_are_rejected(tiny_models, tmp_path):
    copy = tmp_path / "model"
    shutil.copytree(tiny_models["model_dir"], copy)
    with (copy / "model.safetensors").open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ArtifactError, match="checksum"):
        PhonMatchNet(copy, tiny_models["google_dir"], tiny_models["g2p_dir"])


def test_wrong_companion_encoder_is_rejected(tiny_models, tmp_path):
    copy = tmp_path / "model"
    shutil.copytree(tiny_models["model_dir"], copy)
    config = json.loads((copy / "config.json").read_text())
    config["requires"]["google"]["sha256"] = "0" * 64
    (copy / "config.json").write_text(json.dumps(config))
    with pytest.raises(ArtifactError, match="google"):
        PhonMatchNet(copy, tiny_models["google_dir"], tiny_models["g2p_dir"])


def test_swapped_directories_are_rejected(tiny_models):
    with pytest.raises(ArtifactError, match="naptrap-google"):
        PhonMatchNet(tiny_models["model_dir"], tiny_models["g2p_dir"], tiny_models["g2p_dir"])


def test_missing_hub_dependency_has_a_clear_error(tiny_models, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"naptrap\[hub\]"):
        load_model("phonmatchnet", model_dir=tiny_models["model_dir"])


def test_registry_preserves_existing_models_and_rejects_unknown():
    assert "phonmatchnet" in available_models()
    with pytest.raises(ValueError, match="already registered"):
        register_model("phonmatchnet", lambda **kwargs: object())
    with pytest.raises(ValueError, match="Unknown"):
        load_model("nonexistent")


def test_registry_accepts_a_new_family(monkeypatch):
    import naptrap.registry as registry
    monkeypatch.setattr(registry, "_FACTORIES", registry._FACTORIES.copy())
    instance = object()
    register_model("future-model", lambda **kwargs: instance)
    assert load_model("future-model", model_dir="example") is instance


def test_cli_prints_json_and_exit_code(tiny_models, tmp_path, capsys):
    import wave
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)
    args = ["detect", str(path), "hey computer", "--model-dir", str(tiny_models["model_dir"]),
            "--google-dir", str(tiny_models["google_dir"]), "--g2p-dir", str(tiny_models["g2p_dir"])]
    code = main(args)
    result = json.loads(capsys.readouterr().out)
    assert code == (0 if result["accepted"] else 1)
    with pytest.raises(SystemExit) as error:
        main(args[:1] + [str(tmp_path / "missing.wav")] + args[2:])
    assert error.value.code == 2
