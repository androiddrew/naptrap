import array
import errno
import io
import json
import wave

import pytest
import torch

from naptrap import Detection, Keyword, Listener, Prediction, load_model, load_wav
from naptrap.cli import _pcm_chunks, main

RATE = 16000


class Scripted:
    """A stand-in model that returns scripted scores, one per window.

    ``scores`` is one list used for every keyword, or a dict of keyword -> list.
    """

    thresholds = {0.01: 0.5}

    def __init__(self, scores):
        self.scores, self.windows = scores, {}

    def keyword(self, text):
        return Keyword(text, torch.zeros(1, 256))

    def detect(self, waveform, keyword, *, sample_rate=RATE, far=0.01):
        scores = self.scores[keyword.text] if isinstance(self.scores, dict) else self.scores
        window = self.windows.get(keyword.text, 0)
        self.windows[keyword.text] = window + 1
        score = scores[min(window, len(scores) - 1)]
        return Prediction("scripted", "C-KWS", score, 0.5, score >= 0.5, far)

    def predict(self, features, *, far=0.01):
        raise NotImplementedError


def listen(scores, keywords="hey computer", **options):
    """Feed one hop at a time to a listener over a scripted model; returns (listener, detections per window)."""
    listener = Listener(Scripted(scores), keywords, **options)
    count = len(next(iter(scores.values()))) if isinstance(scores, dict) else len(scores)
    heard = [listener.feed(torch.zeros(round(listener.window * RATE)))]
    heard += [listener.feed(torch.zeros(round(listener.hop * RATE))) for _ in range(count - 1)]
    return listener, heard


@pytest.fixture(scope="module")
def tiny(tiny_models):
    return load_model("phonmatchnet", **tiny_models)


def noise(seconds, seed=0):
    return (torch.randn(int(seconds * RATE), generator=torch.Generator().manual_seed(seed)) * 0.1).clamp(-1, 1)


def write_wav(path, samples):
    values = (samples * 32768).round().clamp(-32768, 32767).to(torch.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1), stream.setsampwidth(2), stream.setframerate(RATE)
        stream.writeframes(b"".join(int(v).to_bytes(2, "little", signed=True) for v in values))
    return path


def folders(models):
    return ["--model-dir", str(models["model_dir"]), "--google-dir", str(models["google_dir"]),
            "--g2p-dir", str(models["g2p_dir"])]


def test_scores_equal_detect_on_the_same_windows(tiny):
    audio, keyword = noise(3), tiny.keyword("hey computer")
    listener, seen = Listener(tiny, keyword, threshold=1.0), []
    for start in range(0, audio.numel(), 1777):  # chunks that line up with neither windows nor hops
        before = listener.time
        listener.feed(audio[start:start + 1777])
        if listener.time >= 1.5 and int(listener.time * 4) > int(before * 4):
            seen.append(listener.scores["hey computer"])
    expected = [tiny.detect(audio[end - 24000:end], keyword).score for end in range(24000, audio.numel() + 1, 4000)]
    assert len(expected) == 7 and len(set(expected)) > 1
    assert seen == expected


def test_chunk_sizes_do_not_change_the_result(tiny):
    audio = noise(2.5, seed=1)
    results = []
    for size in (1, 160, 1600, 1777, 40000):
        listener = Listener(tiny, ["hey computer", "stop music"], threshold=0.0, refractory=0.0)
        found = [d for start in range(0, audio.numel(), size) for d in listener.feed(audio[start:start + size])]
        results.append([(d.keyword, d.score, d.end) for d in found])
    assert results[0] and all(result == results[0] for result in results)


def test_nothing_is_scored_before_a_full_window(tiny):
    listener = Listener(tiny, "hey computer", threshold=0.0)
    assert listener.feed(noise(1.49)) == [] and listener.scores == {}
    assert listener.feed(torch.zeros(0)) == []
    assert len(listener.feed(noise(0.01))) == 1
    assert listener.time == pytest.approx(1.5)


def test_several_keywords_share_one_stream(tiny):
    listener = Listener(tiny, ["hey computer", "stop music"], threshold=0.0)
    found = listener.feed(noise(1.5))
    assert [d.keyword for d in found] == ["hey computer", "stop music"] == listener.keywords
    scores = {text: tiny.detect(noise(1.5), text).score for text in listener.keywords}
    assert listener.scores == scores and scores["hey computer"] != scores["stop music"]


def test_the_same_keyword_twice_is_listened_for_once(tiny):
    listener = Listener(tiny, ["Hey Computer", "hey computer", tiny.keyword("hey computer")], threshold=0.0)
    assert listener.keywords == ["hey computer"]
    assert len(listener.feed(noise(1.5))) == 1


def test_one_utterance_gives_one_detection():
    listener, heard = listen([0.1, 0.9, 0.95, 0.9, 0.1, 0.1])
    assert [len(h) for h in heard] == [0, 1, 0, 0, 0, 0]
    detection = heard[1][0]
    assert isinstance(detection, Detection) and detection.keyword == "hey computer"
    assert (detection.start, detection.end, detection.score, detection.threshold) == (0.25, 1.75, 0.9, 0.5)
    assert json.loads(json.dumps(detection.to_dict()))["end"] == 1.75
    assert (listener.window, listener.hop) == (1.5, 0.25)


def test_the_keyword_is_reported_again_after_the_refractory_time():
    scores = [0.9] + [0.1] * 7 + [0.9, 0.1]
    assert [len(h) for h in listen(scores)[1]] == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]          # 2 s later
    assert [len(h) for h in listen(scores, refractory=5)[1]] == [1] + [0] * 9
    assert [len(h) for h in listen([0.9, 0.9, 0.9], refractory=0)[1]] == [1, 1, 1]


@pytest.mark.parametrize("hop, refractory, gap", [(0.2, 0.6, 3), (0.3, 0.9, 3), (0.05, 0.15, 3), (0.02, 0.1, 5)])
def test_refractory_is_exact_for_any_hop(hop, refractory, gap):
    heard = listen([0.9] * 20, hop=hop, refractory=refractory)[1]
    fired = [n for n, h in enumerate(heard) if h]
    assert fired[0] == 0 and {b - a for a, b in zip(fired, fired[1:])} == {gap}


def test_a_held_score_is_reported_again_with_its_own_peak():
    listener, heard = listen([0.99] + [0.6] * 9, patience=2, refractory=1)
    assert [(n, h[0].score) for n, h in enumerate(heard) if h] == [(1, 0.99), (5, 0.6), (9, 0.6)]


def test_patience_needs_windows_in_a_row():
    listener, heard = listen([0.9, 0.1, 0.9, 0.6, 0.99, 0.1], patience=2)
    assert [len(h) for h in heard] == [0, 0, 0, 1, 0, 0]
    assert heard[3][0].score == 0.9  # the best score in the run so far


def test_a_broken_run_forgets_its_peak():
    listener, heard = listen([0.99, 0.1, 0.6, 0.7], patience=2)
    assert [h[0].score for h in heard if h] == [0.7]


def test_keywords_keep_their_own_patience_and_refractory():
    scores = {"hey computer": [0.9, 0.1, 0.9, 0.1, 0.9, 0.9], "stop music": [0.1, 0.9, 0.1, 0.9, 0.1, 0.1]}
    listener, heard = listen(scores, ["hey computer", "stop music"], patience=2)
    assert [[d.keyword for d in h] for h in heard] == [[], [], [], [], [], ["hey computer"]]  # no run of two is shared
    scores = {"hey computer": [0.9, 0.1, 0.1, 0.1], "stop music": [0.1, 0.8, 0.1, 0.1]}
    listener, heard = listen(scores, ["hey computer", "stop music"])
    assert [[(d.keyword, d.score) for d in h] for h in heard] == [[("hey computer", 0.9)], [("stop music", 0.8)], [], []]
    assert listener.scores == {"hey computer": 0.1, "stop music": 0.1}


def test_threshold_defaults_to_the_calibrated_operating_point(tiny):
    assert Listener(tiny, "hey computer").threshold == tiny.thresholds[0.01]
    assert Listener(tiny, "hey computer", far=0.1).threshold == tiny.thresholds[0.1]
    assert Listener(tiny, "hey computer", threshold=0.25).threshold == 0.25
    with pytest.raises(ValueError, match="calibrated operating points"):
        Listener(tiny, "hey computer", far=0.5)
    with pytest.raises(ValueError, match="calibrated operating points"):  # far is still used to score each window
        Listener(tiny, "hey computer", threshold=0.25, far=0.5)


def test_reset_starts_a_new_stream(tiny):
    listener = Listener(tiny, "hey computer", threshold=0.0)
    assert len(listener.feed(noise(1.5))) == 1
    listener.reset()
    assert (listener.time, listener.scores) == (0.0, {})
    assert listener.feed(noise(1.0)) == [] and len(listener.feed(noise(0.5))) == 1


def test_pcm16_bytes_match_float_samples(tiny):
    values = (noise(1.5) * 32768).round().clamp(-32768, 32767).to(torch.int16)
    data = b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)
    expected = Listener(tiny, "hey computer", threshold=0.0).feed(values.to(torch.float32) / 32768)[0].score
    for form in (data, bytearray(data), memoryview(data)):
        assert Listener(tiny, "hey computer", threshold=0.0).feed_pcm16(form)[0].score == expected
    listener = Listener(tiny, "hey computer", threshold=0.0)
    assert listener.feed_pcm16(b"") == []
    floats = array.array("f", [0.0, 0.0])  # the right number of bytes, but not bytes: must not be read as PCM
    for bad in (b"\x00", "0000", None, [0, 0], floats):
        with pytest.raises(ValueError):
            listener.feed_pcm16(bad)
    assert listener.time == 0.0


@pytest.mark.parametrize("samples", [
    torch.zeros(100, dtype=torch.float64), torch.zeros(10, 2), torch.ones(100) * 2,
    torch.full((100,), float("nan")), torch.zeros(100, dtype=torch.int16),
])
def test_bad_samples_are_rejected(tiny, samples):
    with pytest.raises(ValueError):
        Listener(tiny, "hey computer").feed(samples)


@pytest.mark.parametrize("options", [
    {"hop": 0}, {"window": 0}, {"window": 0.4}, {"window": 2.5}, {"window": float("inf")}, {"hop": float("nan")},
    {"patience": 0}, {"patience": 1.5}, {"refractory": -1}, {"refractory": float("inf")}, {"threshold": 1.5},
])
def test_bad_settings_are_rejected(tiny, options):
    with pytest.raises(ValueError):
        Listener(tiny, "hey computer", **options)
    with pytest.raises(ValueError):
        Listener(tiny, [])


def test_window_may_be_any_clip_length_the_model_accepts(tiny):
    for window in (0.5, 2.0):
        listener = Listener(tiny, "hey computer", threshold=0.0, window=window, hop=3.0)  # a hop longer than the window
        found = listener.feed(noise(window + 3.0))
        assert [d.end for d in found] == [window, window + 3.0]
        assert found[0].score == tiny.detect(noise(window + 3.0)[:round(window * RATE)], "hey computer").score


def test_a_model_without_shared_features_needs_a_threshold(tiny_zp_kws, monkeypatch):
    monkeypatch.setenv("NAPTRAP_EXPERIMENTAL", "zp-kws")
    model = load_model("zp-kws", **tiny_zp_kws)
    with pytest.raises(ValueError, match="pass threshold="):
        Listener(model, "hey computer")
    with pytest.raises(ValueError, match="calibrated operating points"):
        Listener(model, "hey computer", threshold=0.5, far=0.5)
    listener = Listener(model, ["hey computer", "stop music"], threshold=0.0)
    found = listener.feed(noise(1.5))
    assert [d.keyword for d in found] == ["hey computer", "stop music"]
    assert listener.scores == {text: model.detect(noise(1.5), text).score for text in listener.keywords}


def test_listen_command_reads_a_wav_file(tiny_models, tmp_path, capsys):
    path = write_wav(tmp_path / "stream.wav", noise(2))
    options = ["listen", "hey computer", "--input", str(path)] + folders(tiny_models)
    assert main(options + ["--threshold", "0", "--refractory", "0"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["end"] for line in lines] == [1.5, 1.75, 2.0] and lines[0]["keyword"] == "hey computer"
    assert main(options + ["--threshold", "1"]) == 1
    assert capsys.readouterr() == ("", "")


def test_listen_command_reads_pcm_from_standard_input(tiny_models, tmp_path, capsys, monkeypatch):
    values = (noise(2) * 32768).round().clamp(-32768, 32767).to(torch.int16)
    data = b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)
    path = write_wav(tmp_path / "stream.wav", noise(2))
    options = ["listen", "hey computer", "--threshold", "0", "--refractory", "0"] + folders(tiny_models)
    assert main(options + ["--input", str(path)]) == 0
    from_file = capsys.readouterr().out

    class Stdin:
        buffer = io.BytesIO(data + b"\x7f")  # a stray last byte is dropped

        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", Stdin())
    assert main(options) == 0
    assert capsys.readouterr().out == from_file and from_file.count("\n") == 3


def test_pcm_chunks_keep_samples_whole():
    class Reader:
        """Returns the pieces it was given, whatever size was asked for."""

        def __init__(self, pieces):
            self.pieces = list(pieces)

        def read(self, size):
            return self.pieces.pop(0) if self.pieces else b""

    assert list(_pcm_chunks(Reader([b"\x01", b"\x02\x03", b"\x04\x05\x06", b"\x07"]))) == [
        b"\x01\x02", b"\x03\x04\x05\x06"]  # a split sample is joined; the stray last byte is dropped
    assert list(_pcm_chunks(io.BytesIO(bytes(range(7))), samples=2)) == [b"\x00\x01\x02\x03", b"\x04\x05"]
    assert list(_pcm_chunks(io.BytesIO(b""))) == []


def test_listen_command_says_when_nothing_could_be_scored(tiny_models, tmp_path, capsys):
    path = write_wav(tmp_path / "short.wav", noise(1))
    assert main(["listen", "hey computer", "--input", str(path), "--threshold", "0"] + folders(tiny_models)) == 1
    out, err = capsys.readouterr()
    assert out == "" and "before one 1.5 s window" in err


@pytest.mark.parametrize("content", [b"", b"this is not a WAV file at all"])
def test_commands_reject_a_file_that_is_not_a_wav(tiny_models, tmp_path, capsys, content):
    path = tmp_path / "broken.wav"
    path.write_bytes(content)
    for command in (["listen", "hey computer", "--input", str(path)], ["detect", str(path), "hey computer"]):
        with pytest.raises(SystemExit) as stopped:
            main(command + folders(tiny_models))
        assert stopped.value.code == 2
        assert "not a readable WAV file" in capsys.readouterr().err


@pytest.mark.parametrize("options, message", [
    (["--window", "2.5"], "window must be 0.5-2 seconds"),
    (["--threshold", "0", "--far", "0.5"], "calibrated operating points"),
])
def test_listen_command_rejects_bad_settings_before_reading_audio(tiny_models, tmp_path, capsys, options, message):
    path = write_wav(tmp_path / "stream.wav", noise(2))
    with pytest.raises(SystemExit) as stopped:
        main(["listen", "hey computer", "--input", str(path)] + options + folders(tiny_models))
    assert stopped.value.code == 2 and message in capsys.readouterr().err


def test_listen_command_needs_audio_from_somewhere(tiny_models, capsys, monkeypatch):
    class Terminal:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Terminal())
    with pytest.raises(SystemExit) as stopped:
        main(["listen", "hey computer"] + folders(tiny_models))
    assert stopped.value.code == 2 and "--input" in capsys.readouterr().err


def test_listen_command_reports_an_input_error(tiny_models, capsys, monkeypatch):
    class Broken:
        def read(self, size):
            raise OSError(errno.EINVAL, "the input went wrong")  # the code a closed output pipe has on Windows

    class Stdin:
        buffer = Broken()

        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", Stdin())
    with pytest.raises(SystemExit) as stopped:
        main(["listen", "hey computer"] + folders(tiny_models))
    assert stopped.value.code == 2 and "the input went wrong" in capsys.readouterr().err


def test_listen_command_stops_quietly_when_its_reader_leaves(tiny_models, tmp_path, capfd, monkeypatch):
    path = write_wav(tmp_path / "stream.wav", noise(2))

    def closed(*args, **kwargs):
        raise BrokenPipeError

    monkeypatch.setattr("builtins.print", closed)
    monkeypatch.setattr("os.dup2", lambda *args: None)
    options = ["listen", "hey computer", "--input", str(path), "--threshold", "0"] + folders(tiny_models)
    assert main(options) == 0
    assert capfd.readouterr().err == ""


def test_released_model_hears_its_example_once_in_a_stream(real_models):
    model = load_model("phonmatchnet", **real_models)
    said = load_wav(real_models["model_dir"] / "examples" / "said-mister.wav")
    stream = torch.cat([torch.zeros(2 * RATE), said, torch.zeros(3 * RATE)])
    listener = Listener(model, ["said mister", "hey computer"])
    found = [d for start in range(0, stream.numel(), 1600) for d in listener.feed(stream[start:start + 1600])]
    assert [d.keyword for d in found] == ["said mister"]
    assert 2.0 < found[0].end < 2.0 + said.numel() / RATE + 1.5
