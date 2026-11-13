import json
import wave

import pytest
import torch

from naptrap import Detection, Keyword, Listener, Prediction, load_model
from naptrap.cli import main

RATE = 16000


class Scripted:
    """A stand-in model that returns a fixed list of scores, one per window, for every keyword."""

    thresholds = {0.01: 0.5}

    def __init__(self, scores):
        self.scores, self.calls = list(scores), 0

    def keyword(self, text):
        return Keyword(text, torch.zeros(1, 256))

    def detect(self, waveform, keyword, *, sample_rate=RATE, far=0.01):
        score = self.scores[min(self.calls // self.per_window, len(self.scores) - 1)]
        self.calls += 1
        return Prediction("scripted", "C-KWS", score, 0.5, score >= 0.5, far)

    per_window = 1

    def predict(self, features, *, far=0.01):
        raise NotImplementedError


def listen(scores, keywords="hey computer", **options):
    """Feed one hop at a time to a listener over a scripted model; returns (listener, detections per window)."""
    model = Scripted(scores)
    listener = Listener(model, keywords, **options)
    model.per_window = len(listener.keywords)
    heard = [listener.feed(torch.zeros(RATE * 3 // 2))]
    heard += [listener.feed(torch.zeros(RATE // 4)) for _ in scores[1:]]
    return listener, heard


@pytest.fixture(scope="module")
def tiny(tiny_models):
    model = load_model("phonmatchnet", **tiny_models)
    # The fixture's random mel filters can be negative, which only silence survives the logarithm of.
    model.google_encoder.mel_matrix.abs_()
    model._frontend.mel_matrix.abs_()
    return model


def noise(seconds, seed=0):
    return (torch.randn(int(seconds * RATE), generator=torch.Generator().manual_seed(seed)) * 0.1).clamp(-1, 1)


def test_scores_equal_detect_on_the_same_windows(tiny):
    audio, keyword = noise(3), tiny.keyword("hey computer")
    listener, seen = Listener(tiny, keyword, threshold=1.0), []
    for start in range(0, audio.numel(), 1000):  # chunks that line up with neither windows nor hops
        before = listener.scores
        listener.feed(audio[start:start + 1000])
        if listener.scores != before or (not before and listener.scores):
            seen.append(listener.scores["hey computer"])
    expected = [tiny.detect(audio[end - 24000:end], keyword).score for end in range(24000, audio.numel() + 1, 4000)]
    assert len(expected) == 7
    assert seen == pytest.approx(expected, abs=1e-6)


def test_chunk_sizes_do_not_change_the_result(tiny):
    audio = noise(2.5, seed=1)
    results = []
    for size in (1, 160, 1600, 40000):
        listener = Listener(tiny, ["hey computer", "stop music"], threshold=0.0, refractory=0.0)
        found = [d for start in range(0, audio.numel(), size) for d in listener.feed(audio[start:start + size])]
        results.append([(d.keyword, round(d.score, 6), d.end) for d in found])
    assert results[0] and all(result == results[0] for result in results)


def test_nothing_is_scored_before_a_full_window(tiny):
    listener = Listener(tiny, "hey computer", threshold=0.0)
    assert listener.feed(noise(1.49)) == [] and listener.scores == {}
    assert len(listener.feed(noise(0.01))) == 1
    assert listener.time == pytest.approx(1.5)


def test_several_keywords_share_one_stream(tiny):
    listener = Listener(tiny, ["hey computer", "stop music"], threshold=0.0)
    found = listener.feed(noise(1.5))
    assert [d.keyword for d in found] == ["hey computer", "stop music"]
    assert set(listener.scores) == {"hey computer", "stop music"} == set(listener.keywords)
    assert listener.scores["hey computer"] == pytest.approx(tiny.detect(noise(1.5), "hey computer").score, abs=1e-6)


def test_one_utterance_gives_one_detection():
    listener, heard = listen([0.1, 0.9, 0.95, 0.9, 0.1, 0.1])
    assert [len(h) for h in heard] == [0, 1, 0, 0, 0, 0]
    detection = heard[1][0]
    assert isinstance(detection, Detection) and detection.keyword == "hey computer"
    assert (detection.start, detection.end, detection.score, detection.threshold) == (0.25, 1.75, 0.9, 0.5)
    assert json.loads(json.dumps(detection.to_dict()))["end"] == 1.75


def test_the_keyword_is_reported_again_after_the_refractory_time():
    scores = [0.9] + [0.1] * 7 + [0.9, 0.1]
    assert [len(h) for h in listen(scores)[1]] == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]          # 2 s later
    assert [len(h) for h in listen(scores, refractory=5)[1]] == [1] + [0] * 9
    assert [len(h) for h in listen([0.9, 0.9, 0.9], refractory=0)[1]] == [1, 1, 1]


def test_patience_needs_windows_in_a_row():
    listener, heard = listen([0.9, 0.1, 0.9, 0.6, 0.99, 0.1], patience=2)
    assert [len(h) for h in heard] == [0, 0, 0, 1, 0, 0]
    assert heard[3][0].score == 0.9  # the best score in the run so far


def test_threshold_defaults_to_the_calibrated_operating_point(tiny):
    assert Listener(tiny, "hey computer").threshold == tiny.thresholds[0.01]
    assert Listener(tiny, "hey computer", far=0.1).threshold == tiny.thresholds[0.1]
    assert Listener(tiny, "hey computer", threshold=0.25).threshold == 0.25
    with pytest.raises(ValueError):
        Listener(tiny, "hey computer", far=0.5)


def test_reset_starts_a_new_stream(tiny):
    listener = Listener(tiny, "hey computer", threshold=0.0)
    assert len(listener.feed(noise(1.5))) == 1
    listener.reset()
    assert (listener.time, listener.scores) == (0.0, {})
    assert listener.feed(noise(1.0)) == [] and len(listener.feed(noise(0.5))) == 1


def test_pcm16_bytes_match_float_samples(tiny):
    values = (noise(1.5) * 32768).round().clamp(-32768, 32767).to(torch.int16)
    from_bytes = Listener(tiny, "hey computer", threshold=0.0)
    from_floats = Listener(tiny, "hey computer", threshold=0.0)
    data = b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)
    assert from_bytes.feed_pcm16(data)[0].score == from_floats.feed(values.to(torch.float32) / 32768)[0].score
    with pytest.raises(ValueError):
        from_bytes.feed_pcm16(b"\x00")


@pytest.mark.parametrize("samples", [
    torch.zeros(100, dtype=torch.float64), torch.zeros(10, 2), torch.ones(100) * 2,
    torch.full((100,), float("nan")), torch.zeros(100, dtype=torch.int16),
])
def test_bad_samples_are_rejected(tiny, samples):
    with pytest.raises(ValueError):
        Listener(tiny, "hey computer").feed(samples)


@pytest.mark.parametrize("options", [
    {"hop": 0}, {"window": 0}, {"patience": 0}, {"patience": 1.5}, {"refractory": -1}, {"threshold": 1.5},
])
def test_bad_settings_are_rejected(tiny, options):
    with pytest.raises(ValueError):
        Listener(tiny, "hey computer", **options)
    with pytest.raises(ValueError):
        Listener(tiny, [])


def test_listen_command_reads_a_wav_file(tiny_models, tmp_path, capsys):
    path = tmp_path / "stream.wav"
    samples = torch.zeros(2 * RATE, dtype=torch.int16)  # the random-weight fixture only accepts silence
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1), stream.setsampwidth(2), stream.setframerate(RATE)
        stream.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))
    folders = ["--model-dir", str(tiny_models["model_dir"]), "--google-dir", str(tiny_models["google_dir"]),
               "--g2p-dir", str(tiny_models["g2p_dir"])]
    assert main(["listen", "hey computer", "--input", str(path), "--threshold", "0", "--refractory", "0"] + folders) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["end"] for line in lines] == [1.5, 1.75, 2.0] and lines[0]["keyword"] == "hey computer"
    assert main(["listen", "hey computer", "--input", str(path), "--threshold", "1"] + folders) == 1


def test_released_model_hears_its_example_once_in_a_stream(real_models):
    model = load_model("phonmatchnet", **real_models)
    from naptrap import load_wav
    said = load_wav(real_models["model_dir"] / "examples" / "said-mister.wav")
    stream = torch.cat([torch.zeros(2 * RATE), said, torch.zeros(3 * RATE)])
    listener = Listener(model, ["said mister", "hey computer"])
    found = [d for start in range(0, stream.numel(), 1600) for d in listener.feed(stream[start:start + 1600])]
    assert [d.keyword for d in found] == ["said mister"]
    assert 2.0 < found[0].end < 2.0 + said.numel() / RATE + 1.5
