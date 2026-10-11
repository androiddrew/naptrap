"""Optional real-artifact tests; models remain outside the library/repository."""

import os
from pathlib import Path

import numpy as np
import pytest

from naptrap import load_model


GOLDEN = {
    17: (0.9978187084197998, 0.7149204015731812),
    29: (0.9978514909744263, 0.7729189991950989),
    43: (0.9722860455513, 0.5510875582695007),
}


@pytest.fixture(scope="module")
def bundle():
    value = os.environ.get("NAPTRAP_TEST_BUNDLE")
    if not value:
        pytest.skip("Set NAPTRAP_TEST_BUNDLE to run retained real-model fixtures")
    return Path(value).expanduser()


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_original_candidate_score_and_threshold_are_preserved(bundle, seed):
    # Golden values come from the completed original isolated CPU CLI report,
    # not from a new training/calibration run or this implementation itself.
    model = load_model("phonmatchnet", bundle=bundle, seed=seed)
    with np.load(bundle / "examples" / ("said-mister-seed%d.npz" % seed), allow_pickle=False) as f:
        result = model.predict(f)
        repeated = model.predict({key: f[key][0] for key in ("keyword_mel", "google", "text")})
    assert result.score == pytest.approx(GOLDEN[seed][0], abs=2e-6)
    assert result.threshold == GOLDEN[seed][1]
    assert result.accepted and result.task == "C-KWS"
    assert repeated == result
    public_thresholds = model.thresholds
    public_thresholds[.01] = 0
    assert model.thresholds[.01] == GOLDEN[seed][1]


def test_mismatched_clocks_and_unregistered_operating_point_fail(bundle):
    model = load_model(bundle=bundle)
    with np.load(bundle / "examples/said-mister-seed17.npz", allow_pickle=False) as f:
        with pytest.raises(ValueError, match="clocks"):
            model.predict({"keyword_mel": f["keyword_mel"], "google": np.zeros((25, 96), dtype=np.float32), "text": f["text"]})
        with pytest.raises(ValueError, match="far"):
            model.predict(f, far=.05)
