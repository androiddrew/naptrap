"""The experimental ZP-KWS backend: feature flag, plumbing and, if available, released weights."""

import json
import wave

import pytest
import torch
from safetensors.torch import load_file

from naptrap import available_models, load_model, load_wav, register_model
from naptrap.cli import main
from naptrap.models.zp_kws import Enrollment, ZpKws

# Reference outputs of the original training-time pipeline for the bundled
# example: said-mister.wav against "said mister", seed 17.
GOLDEN_LOGIT = 6.887462615966797
GOLDEN_SAME_SPEAKER = 0.3626159727573395
GOLDEN_OTHER_SPEAKER = 0.007688208483159542
GOLDEN_THRESHOLD = 0.2188410758972168


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("NAPTRAP_EXPERIMENTAL", "zp-kws")


def write_wav(path, samples):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * samples)
    return path


def test_hidden_unless_enabled(tiny_zp_kws, monkeypatch, capsys):
    monkeypatch.delenv("NAPTRAP_EXPERIMENTAL", raising=False)
    assert "zp-kws" not in available_models()
    with pytest.raises(ValueError, match="NAPTRAP_EXPERIMENTAL=zp-kws"):
        load_model("zp-kws", **tiny_zp_kws)
    with pytest.raises(SystemExit):
        main(["detect", "--help"])
    assert "--enroll" not in capsys.readouterr().out


def test_name_cannot_be_registered_by_others():
    with pytest.raises(ValueError, match="already registered"):
        register_model("zp-kws", lambda **kwargs: object())


def test_detect_with_and_without_enrollment(tiny_zp_kws, enabled):
    assert "zp-kws" in available_models()
    model = load_model("zp-kws", **tiny_zp_kws)
    assert isinstance(model, ZpKws)
    enrollment = model.enroll(torch.zeros(12000))
    assert isinstance(enrollment, Enrollment) and enrollment.embedding.shape == (192,)
    personal = model.detect(torch.zeros(16000), "hey computer", enrollment=enrollment)
    assert personal.task == "TO-KWS" and personal.threshold == 0.2
    assert set(personal.details) == {"keyword_logit", "keyword_probability", "speaker_cosine", "speaker_probability"}
    assert personal.score == pytest.approx(personal.details["keyword_probability"]
                                           * personal.details["speaker_probability"])
    content = model.detect(torch.zeros(16000), model.keyword("hey computer"), far=0.1)
    assert content.task == "C-KWS" and content.threshold == 0.5
    assert content.score == personal.details["keyword_probability"]
    assert model.thresholds == {"TO-KWS": {0.01: 0.2, 0.1: 0.02}, "C-KWS": {0.01: 0.9, 0.1: 0.5}}


def test_invalid_inputs_are_rejected(tiny_zp_kws, enabled):
    model = load_model("zp-kws", **tiny_zp_kws)
    with pytest.raises(ValueError, match="0.5-2 seconds"):
        model.enroll(torch.zeros(7999))
    with pytest.raises(ValueError, match="far"):
        model.detect(torch.zeros(16000), "stop", far=0.05)
    features = {"keyword_mel": torch.zeros(100, 40), "google": torch.zeros(12, 96), "text": torch.zeros(3, 256)}
    enrollment = model.enroll_features(torch.zeros(80, 101))
    with pytest.raises(ValueError, match="speaker_mel"):
        model.predict(features, enrollment=enrollment)
    with pytest.raises(ValueError, match="different clip lengths"):
        model.predict({**features, "speaker_mel": torch.zeros(80, 120)}, enrollment=enrollment)
    with pytest.raises(ValueError, match=r"\[80,frames\]"):
        model.enroll_features(torch.zeros(101, 80))
    with pytest.raises(TypeError, match="enroll"):
        model.predict({**features, "speaker_mel": torch.zeros(80, 101)}, enrollment=torch.zeros(192))


def test_cli_enroll(tiny_zp_kws, enabled, tmp_path, capsys):
    clip = write_wav(tmp_path / "clip.wav", 16000)
    speaker = write_wav(tmp_path / "speaker.wav", 12000)
    dirs = ["--model-dir", str(tiny_zp_kws["model_dir"]), "--google-dir", str(tiny_zp_kws["google_dir"]),
            "--g2p-dir", str(tiny_zp_kws["g2p_dir"])]
    code = main(["detect", str(clip), "hey computer", "--model", "zp-kws", "--enroll", str(speaker)] + dirs)
    result = json.loads(capsys.readouterr().out)
    assert result["task"] == "TO-KWS" and code == (0 if result["accepted"] else 1)
    dirs[1] = str(tiny_zp_kws["model_dir"].parent / "phonmatchnet")
    with pytest.raises(SystemExit) as error:
        main(["detect", str(clip), "hey computer", "--enroll", str(speaker)] + dirs)
    assert error.value.code == 2


@pytest.fixture(scope="module")
def released(real_zp_kws):
    model = ZpKws.from_pretrained(**real_zp_kws)
    directory = real_zp_kws["model_dir"] / "examples"
    clips = {name: load_wav(directory / f"{name}.wav")
             for name in ("said-mister", "enroll-same-speaker", "enroll-other-speaker")}
    return model, clips, load_file(str(directory / "said-mister.safetensors"))


def test_reference_features_reproduce_the_original_scores(released):
    model, _, features = released
    enrollment = model.enroll_features(features["enrollment_speaker_mel"])
    query = {key: features[key] for key in ("keyword_mel", "google", "text")}
    result = model.predict({**query, "speaker_mel": features["query_speaker_mel"]}, enrollment=enrollment)
    assert result.details["keyword_logit"] == pytest.approx(GOLDEN_LOGIT, abs=1e-5)
    assert result.score == pytest.approx(GOLDEN_SAME_SPEAKER, abs=2e-6)
    assert result.threshold == GOLDEN_THRESHOLD and result.accepted
    assert model.predict(query).task == "C-KWS"


def test_audio_pipeline_accepts_the_speaker_and_rejects_another(released):
    model, clips, features = released
    torch.testing.assert_close(model._model.speaker.frontend(clips["said-mister"])[0],
                               features["query_speaker_mel"], rtol=0, atol=1e-3)
    same = model.detect(clips["said-mister"], "said mister", enrollment=model.enroll(clips["enroll-same-speaker"]))
    assert same.score == pytest.approx(GOLDEN_SAME_SPEAKER, abs=2e-5) and same.accepted
    other = model.detect(clips["said-mister"], "said mister", enrollment=model.enroll(clips["enroll-other-speaker"]))
    assert other.score == pytest.approx(GOLDEN_OTHER_SPEAKER, abs=2e-5) and not other.accepted
