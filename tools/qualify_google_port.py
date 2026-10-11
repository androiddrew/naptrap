"""Compare a TF-free PyTorch Google port to retained TF/cached feature fixtures."""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

from naptrap import load_model
from naptrap.google import GoogleSpeechEmbedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    config = json.loads(args.config.read_text())
    source, converted, output = (root / config[k] for k in ("reference_directory", "converted_directory", "output_directory"))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    reference = json.loads((source / "report.json").read_text())
    if reference["stage"] != "reference_captured":
        raise ValueError("Reference incomplete")
    encoder = GoogleSpeechEmbedding.from_directory(converted)
    diagnostics = []
    features_by_crop = {}
    for row in reference["examples"]:
        data = np.load(source / (row["name"] + ".npz"), allow_pickle=False)
        wave = torch.from_numpy(data["waveform"].copy())
        with torch.no_grad():
            mel = encoder.log_mel(wave)
            actual = encoder(wave).cpu().numpy()
            same_frontend = encoder.encode_log_mel(torch.from_numpy(data["log_mel"])).cpu().numpy()
        expected = data["features"]
        item = {"name": row["name"], "shape": list(actual.shape),
                "maximum_feature_error": float(np.max(np.abs(actual - expected))),
                "same_logmel_cnn_maximum_error": float(np.max(np.abs(same_frontend - expected))),
                "maximum_log_mel_error": float(np.max(np.abs(mel.numpy() - data["log_mel"]))) }
        diagnostics.append(item)
        (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        np.testing.assert_allclose(same_frontend, expected, rtol=config["feature_relative_tolerance"], atol=config["feature_absolute_tolerance"])
        if row["name"].startswith("tone-") and "tonal_fft_global_relative_tolerance" in config:
            scaled = item["maximum_feature_error"] / max(1., float(np.abs(expected).max()))
            if scaled > config["tonal_fft_global_relative_tolerance"]:
                raise ValueError("Tonal FFT numerical error exceeds the registered global bound")
            item["tonal_global_relative_error"] = scaled
        else:
            np.testing.assert_allclose(actual, expected, rtol=config["feature_relative_tolerance"], atol=config["feature_absolute_tolerance"])
        if "cached_source_features" in data:
            cached = data["cached_source_features"]
            np.testing.assert_allclose(actual, cached, rtol=config["feature_relative_tolerance"], atol=config["feature_absolute_tolerance"])
            item["maximum_original_machine_feature_error"] = float(np.max(np.abs(actual - cached)))
            features_by_crop[row["name"][len("speech-"):]] = (wave, actual, cached)
    manifest = json.loads((root / "data/derived/development-clean460-v1/manifest.json").read_text())
    text_db = sqlite3.connect((root / "data/cache/keyword-clean460-text-v1/features.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)
    downstream = []
    for seed in (17, 29, 43):
        model = load_model(bundle=root / "releases/python-onnx-v1", seed=seed)
        for crop_id, (wave, actual, expected) in features_by_crop.items():
            for match in (0, 1):
                candidates = sorted((r for r in manifest["trials"] if r["split"] == "dev-select" and r["query_crop_id"] == crop_id and r["keyword_match"] == match), key=lambda r: r["trial_id"])
                if not candidates:
                    continue
                trial = candidates[0]
                target = manifest["query_crops"][trial["keyword_crop_id"]]["text"]
                value = text_db.execute("SELECT valid_length,data FROM features WHERE text=?", (target,)).fetchone()
                text = np.frombuffer(value[1], dtype="<f4").copy().reshape(value[0], 256)
                maximum = 0.
                for far in (.01, .1):
                    old = model.predict_audio(wave, google=expected, text=text, far=far)
                    new = model.predict_audio(wave, google=actual, text=text, far=far)
                    error = abs(old.score - new.score)
                    if error > config["probability_absolute_tolerance"] or old.accepted != new.accepted:
                        raise ValueError("Downstream frozen model/decision parity failed")
                    maximum = max(maximum, error)
                downstream.append({"seed": seed, "crop_id": crop_id, "keyword": target, "keyword_match": match, "max_probability_error": maximum})
        with np.load(root / "releases/python-onnx-v1/examples/said-mister-seed17.npz") as example:
            stress_text = example["text"].copy()
        for row in reference["examples"]:
            if row["name"].startswith("speech-"):
                continue
            data = np.load(source / (row["name"] + ".npz"), allow_pickle=False)
            wave = torch.from_numpy(data["waveform"].copy())
            actual = encoder(wave)
            maximum = 0.
            for far in (.01, .1):
                old = model.predict_audio(wave, google=data["features"], text=stress_text, far=far)
                new = model.predict_audio(wave, google=actual, text=stress_text, far=far)
                error = abs(old.score - new.score)
                if error > config["probability_absolute_tolerance"] or old.accepted != new.accepted:
                    raise ValueError("Stress fixture downstream parity failed")
                maximum = max(maximum, error)
            downstream.append({"seed": seed, "stress": row["name"], "keyword": "said mister", "max_probability_error": maximum})
    text_db.close()
    if any(name.startswith("tensorflow") for name in sys.modules):
        raise ValueError("TensorFlow leaked into inference runtime")
    report = {"stage": "passed", "torch": str(torch.__version__), "python": sys.version, "numpy": np.__version__,
              "reference_tensorflow": reference["tensorflow"], "examples": len(diagnostics), "features": diagnostics,
              "maximum_feature_error": max(r["maximum_feature_error"] for r in diagnostics),
              "downstream_cases": len(downstream), "maximum_probability_error": max(r["max_probability_error"] for r in downstream),
              "threshold_decisions": "exact", "tensorflow_imported": False, "model_fitting": False, "final_evaluation": False,
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "converted_model_sha256": hashlib.sha256((converted / "model.pt").read_bytes()).hexdigest()}
    report["runtime_source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "naptrap/src/naptrap").rglob("*.py"))}
    (output / "downstream.json").write_text(json.dumps(downstream, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
