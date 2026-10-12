"""Load and verify a model artifact directory: ``config.json`` + ``model.safetensors``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

WEIGHTS = "model.safetensors"


class ArtifactError(ValueError):
    """A model artifact is missing, has the wrong format or fails its checksum."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(directory: str | Path, expected_format: str) -> Tuple[Path, Dict[str, Any]]:
    root = Path(directory).expanduser().resolve()
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read config.json in {root}") from error
    if not isinstance(config, dict) or config.get("format") != expected_format:
        raise ArtifactError(f"{root} is not a {expected_format} artifact")
    return root, config


def load_artifact(directory: str | Path, expected_format: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Return ``(config, state_dict)`` after checking format and weight checksum."""
    from safetensors.torch import load_file

    root, config = read_config(directory, expected_format)
    path = root / WEIGHTS
    if not path.is_file():
        raise ArtifactError(f"Missing {WEIGHTS} in {root}")
    expected = config.get("sha256", {}).get(WEIGHTS)
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ArtifactError(f"{WEIGHTS} checksum mismatch in {root}")
    return config, load_file(str(path), device="cpu")
