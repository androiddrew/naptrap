"""Read the published candidate manifest and verify artifacts before loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class BundleError(ValueError):
    """A model bundle is incomplete, incompatible or fails its checksum."""


class ModelBundle:
    def __init__(self, directory: str | Path):
        self.root = Path(directory).expanduser().resolve()
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise BundleError(f"Cannot read model manifest in {self.root}") from error
        if not isinstance(self.manifest, dict) or self.manifest.get("release") != "python-onnx-v1" or not isinstance(self.manifest.get("files"), dict):
            raise BundleError("Unsupported model bundle; expected python-onnx-v1")

    def checked_file(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise BundleError("Artifact path leaves the model bundle") from error
        record = self.manifest["files"].get(relative)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise BundleError(f"Artifact is missing from the manifest: {relative}")
        if not path.is_file():
            raise BundleError(f"Missing model artifact: {relative}")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != record["sha256"] or path.stat().st_size != record.get("bytes"):
            raise BundleError(f"Artifact checksum/size mismatch: {relative}")
        return path

    def checked_json(self, relative: str) -> dict:
        try:
            value = json.loads(self.checked_file(relative).read_text())
        except json.JSONDecodeError as error:
            raise BundleError(f"Invalid JSON artifact: {relative}") from error
        if not isinstance(value, dict):
            raise BundleError(f"JSON artifact must be an object: {relative}")
        return value
