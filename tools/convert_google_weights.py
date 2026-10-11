"""Map the pinned Google checkpoint to native PyTorch state; no training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from naptrap.google import FORMAT, PARENT_SHA256, GoogleSpeechEmbedding


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.reference / "report.json").read_text())
    if report["stage"] != "reference_captured" or report["source"]["archive_sha256"] != PARENT_SHA256:
        raise ValueError("Authoritative Google reference required")
    for file, field in (("variables.npz", "variable_npz_sha256"), ("frontend.npz", "frontend_npz_sha256")):
        if digest(args.reference / file) != report[field]:
            raise ValueError("Conversion input changed")
    args.output.mkdir(parents=True, exist_ok=False)
    variables = np.load(args.reference / "variables.npz", allow_pickle=False)
    frontend = np.load(args.reference / "frontend.npz", allow_pickle=False)
    model = GoogleSpeechEmbedding()
    state = model.state_dict()
    prefix = "embeddings/state/MudpuppyLite/"
    used = {prefix + "ZeroMeanConv2d/weights"}
    state["window"] = torch.from_numpy(frontend["window"].copy())
    state["mel_matrix"] = torch.from_numpy(frontend["mel_matrix"].copy())
    state["initial.weight"] = torch.from_numpy(frontend["initial_centered_hwio"].transpose(3, 2, 0, 1).copy())
    for index in range(19):
        name = prefix + ("Conv" if index == 0 else "Conv_" + str(index)) + "/weights"
        target = f"convs.{index}.weight" if index < 18 else "output.weight"
        state[target] = torch.from_numpy(variables[name].transpose(3, 2, 0, 1).copy())
        used.add(name)
    for index in range(19):
        scope = prefix + ("BatchNorm" if index == 0 else "BatchNorm_" + str(index))
        for old, new in (("gamma", "weight"), ("beta", "bias"), ("moving_mean", "running_mean"), ("moving_variance", "running_var")):
            name = scope + "/" + old
            state[f"norms.{index}.{new}"] = torch.from_numpy(variables[name].copy())
            used.add(name)
    if used != set(variables.files):
        raise ValueError(f"Unconsumed or missing Google variables: {used.symmetric_difference(variables.files)}")
    model.load_state_dict(state, strict=True)
    path = args.output / "model.pt"
    torch.save({"format": FORMAT, "parent_archive_sha256": PARENT_SHA256, "model": model.state_dict()}, path)
    manifest = {"format": FORMAT, "parent_archive_sha256": PARENT_SHA256, "model_sha256": digest(path),
                "license": "Apache-2.0", "publisher": "Google", "original": "google/speech_embedding/1",
                "source": "https://www.kaggle.com/models/google/speech-embedding/TensorFlow1/speech-embedding/1",
                "modifications": ["independent PyTorch graph implementation", "HWIO kernels transposed to OIHW",
                                  "initial spatial-mean-subtracted kernel materialized exactly as original TF inference", "original frontend constants retained"],
                "source_variable_count": len(used), "reference_report_sha256": digest(args.reference / "report.json"),
                "parameters": sum(p.numel() for p in model.parameters()), "model_fitting": False,
                "preprocessing": {"sample_rate": 16000, "leading_zeros": 11120, "frames": "floor(samples/1280)",
                                  "stft": "400-sample Hann, hop160, FFT512, no centering, magnitude spectrum", "mel": "32 bins, natural log(mel+0.001)"}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
