"""One-time TensorFlow reference/constant extraction for the PyTorch port."""

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saved_model_pb2


def constant_value(graph_def, name):
    """Evaluate the original constant-only subgraph, without map control edges."""
    nodes = {n.name: n for n in graph_def.node}
    selected = set()
    def add(key):
        key = key.split(":")[0]
        if key in selected:
            return
        node = nodes[key]
        if node.op in ("Placeholder", "ReadVariableOp", "VarHandleOp"):
            raise ValueError("Requested frontend constant depends on runtime state: " + key)
        selected.add(key)
        for parent in node.input:
            if not parent.startswith("^"):
                add(parent)
    add(name)
    subset = graph_pb2.GraphDef()
    for node in graph_def.node:
        if node.name in selected:
            copied = subset.node.add()
            copied.CopyFrom(node)
            inputs = [v for v in copied.input if not v.startswith("^")]
            del copied.input[:]
            copied.input.extend(inputs)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(subset, name="")
        with tf.compat.v1.Session(graph=graph) as session:
            return session.run(graph.get_tensor_by_name(name + ":0"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = root / "data/artifacts/google-speech-embedding"
    identity = json.loads((source / ".source.json").read_text())
    for name, expected in identity["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Original artifact changed")
    saved = saved_model_pb2.SavedModel()
    saved.ParseFromString((source / "saved_model.pb").read_bytes())
    meta = next(m for m in saved.meta_graphs if not m.meta_info_def.tags)
    window = constant_value(meta.graph_def, "map/while/stft/hann_window/sub_2")
    mel = constant_value(meta.graph_def, "map/while/linear_to_mel_weight_matrix")
    reader = tf.train.load_checkpoint(str(source / "variables/variables"))
    weights = {k: reader.get_tensor(k) for k in sorted(reader.get_variable_to_shape_map())}
    key = "embeddings/state/MudpuppyLite/ZeroMeanConv2d/weights"
    initial = tf.convert_to_tensor(weights[key])
    centered = (initial - tf.reduce_mean(initial, axis=[0, 1], keepdims=True)).numpy()
    np.savez(output / "frontend.npz", window=window, mel_matrix=mel, initial_centered_hwio=centered)
    np.savez(output / "variables.npz", **weights)
    manifest = json.loads((root / "data/derived/development-clean460-v1/manifest.json").read_text())
    crops = sorted((r for r in manifest["query_crops"].values() if r["split"] == "dev-select"), key=lambda r: (r["num_samples"], r["crop_id"]))
    positions = np.unique(np.linspace(0, len(crops) - 1, 32, dtype=int))
    db = sqlite3.connect((root / "data/cache/keyword-clean460-google-v1/features.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)
    examples = []
    for i in positions:
        crop = crops[i]
        row = db.execute("SELECT pcm16,data,valid_length,sha256 FROM features WHERE crop_id=?", (crop["crop_id"],)).fetchone()
        if hashlib.sha256(row[0]).hexdigest() != crop["audio_sha256"] or hashlib.sha256(row[1]).hexdigest() != row[3]:
            raise ValueError("Retained feature/crop identity changed")
        wave = np.frombuffer(row[0], dtype="<i2").astype(np.float32) / 32768
        cached = np.frombuffer(row[1], dtype="<f4").copy().reshape(row[2], 96)
        examples.append(("speech-" + crop["crop_id"], wave, cached))
    db.close()
    rng = np.random.default_rng(20261011)
    for length in (8000, 8001, 8960, 16000, 31999, 32000):
        examples.append((f"silence-{length}", np.zeros(length, dtype=np.float32), None))
    for length in (8000, 16000, 32000):
        noise = (rng.standard_normal(length) * .08).clip(-1, 1).astype(np.float32)
        tone = (.15 * np.sin(np.arange(length, dtype=np.float32) * np.float32(2 * np.pi * 440 / 16000))).astype(np.float32)
        examples += [(f"noise-{length}", noise, None), (f"tone-{length}", tone, None)]
    impulse = np.zeros(16000, dtype=np.float32)
    impulse[0], impulse[7999], impulse[-1] = .9, -.9, .9
    examples.append(("impulses-16000", impulse, None))
    configuration = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=1, device_count={"GPU": 0})
    records = []
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session(graph=graph, config=configuration) as session:
            tf.compat.v1.saved_model.loader.load(session, [], str(source))
            inputs = graph.get_tensor_by_name(meta.signature_def["default"].inputs["default"].name)
            endpoint = graph.get_tensor_by_name(meta.signature_def["default"].outputs["default"].name)
            logmel = graph.get_tensor_by_name("Squeeze:0")
            for name, wave, cached in examples:
                features, mel_values = session.run([endpoint, logmel], {inputs: np.pad(wave, (11120, 0))[None]})
                features = features[0, :, 0, :]
                if features.shape != (len(wave) // 1280, 96) or not np.isfinite(features).all():
                    raise ValueError("Reference feature shape/finite failure")
                arrays = {"waveform": wave, "features": features, "log_mel": mel_values[0]}
                if cached is not None:
                    arrays["cached_source_features"] = cached
                np.savez(output / (name + ".npz"), **arrays)
                records.append({"name": name, "samples": len(wave), "frames": len(features),
                                "cached_source_max_abs": float(np.max(np.abs(cached - features))) if cached is not None else None})
    report = {"stage": "reference_captured", "tensorflow": tf.__version__, "numpy": np.__version__, "source": identity,
              "role": "32 metadata-selected dev-select query crops plus deterministic engineering stress fixtures",
              "examples": records, "variables": len(weights), "window_shape": list(window.shape), "mel_shape": list(mel.shape),
              "variable_npz_sha256": hashlib.sha256((output / "variables.npz").read_bytes()).hexdigest(),
              "frontend_npz_sha256": hashlib.sha256((output / "frontend.npz").read_bytes()).hexdigest(),
              "model_fitting": False, "final_evaluation": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("examples", "source")}, indent=2))


if __name__ == "__main__":
    main()
