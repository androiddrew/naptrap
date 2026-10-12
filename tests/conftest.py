import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from naptrap import G2PTextEncoder, GoogleSpeechEmbedding
from naptrap.google import FORMAT as GOOGLE_FORMAT, PARENT_SHA256
from naptrap.models.network import PhonMatchNetwork
from naptrap.models.phonmatchnet import FORMAT as PHONMATCHNET_FORMAT
from naptrap.text import FORMAT as G2P_FORMAT

REPOS = ("naptrap-phonmatchnet-en", "google-speech-embedding-pytorch", "g2p-en-decoder-pytorch")


def write_artifact(directory: Path, module: torch.nn.Module, config: dict) -> str:
    """Save a module with small random weights as config.json + model.safetensors."""
    directory.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    state = {}
    for key, value in module.state_dict().items():
        if value.is_floating_point():
            value = torch.randn(value.shape, generator=generator) * 0.1
            if "running_var" in key:
                value = value.abs() + 1
        state[key] = value.contiguous()
    weights = directory / "model.safetensors"
    save_file(state, str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    (directory / "config.json").write_text(json.dumps({**config, "sha256": {"model.safetensors": digest}}))
    return digest


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """Random-weight artifacts with the real layouts; useful only for plumbing tests."""
    root = tmp_path_factory.mktemp("models")
    google = write_artifact(root / "google", GoogleSpeechEmbedding(),
                            {"format": GOOGLE_FORMAT, "source": {"archive_sha256": PARENT_SHA256}})
    g2p = write_artifact(root / "g2p", G2PTextEncoder(), {"format": G2P_FORMAT, "hidden_size": 256, "phonemes": 74})
    write_artifact(root / "phonmatchnet", PhonMatchNetwork(), {
        "format": PHONMATCHNET_FORMAT,
        "thresholds": {"0.01": 0.9, "0.1": 0.5},
        "requires": {"google": {"repo_id": "x/google", "revision": "v1", "sha256": google},
                     "g2p": {"repo_id": "x/g2p", "revision": "v1", "sha256": g2p}},
    })
    return {"model_dir": root / "phonmatchnet", "google_dir": root / "google", "g2p_dir": root / "g2p"}


@pytest.fixture(scope="session")
def real_models():
    value = os.environ.get("NAPTRAP_TEST_MODELS")
    if not value:
        pytest.skip("Set NAPTRAP_TEST_MODELS to a directory holding the released model repositories")
    root = Path(value).expanduser()
    model_dir, google_dir, g2p_dir = (root / name for name in REPOS)
    return {"model_dir": model_dir, "google_dir": google_dir, "g2p_dir": g2p_dir}
