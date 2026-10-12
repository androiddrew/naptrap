"""Optional Hugging Face Hub downloads (``pip install 'naptrap[hub]'``)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_REVISION = "v1"
_FILES = ["config.json", "model.safetensors", "README.md", "LICENSE", "NOTICE"]


def download(repo_id: str, revision: str = DEFAULT_REVISION, cache_dir: Optional[str] = None) -> Path:
    """Download one model repository's inference files and return its local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "Downloading models requires huggingface_hub: pip install 'naptrap[hub]'. "
            "Alternatively, download the model directories yourself and pass their paths."
        ) from error
    return Path(snapshot_download(repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=_FILES))
