"""Offline graph/weight inspection; TensorFlow is confined to this tooling."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((args.source / ".source.json").read_text())
    for name, expected in identity["files"].items():
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source Google artifact changed: " + name)
    saved = saved_model_pb2.SavedModel()
    saved.ParseFromString((args.source / "saved_model.pb").read_bytes())
    graph = next(g for g in saved.meta_graphs if not g.meta_info_def.tags)
    (args.output / "graph.pb").write_bytes(graph.graph_def.SerializeToString())
    ops = []
    constants = {}
    for node in graph.graph_def.node:
        attrs = {}
        for name, value in node.attr.items():
            field = value.WhichOneof("value")
            if field in ("s", "i", "f", "b"):
                v = getattr(value, field)
                attrs[name] = v.decode(errors="replace") if isinstance(v, bytes) else v
            elif field == "list":
                attrs[name] = {k: list(getattr(value.list, k)) for k in ("i", "f", "b") if len(getattr(value.list, k))}
            elif field == "shape":
                attrs[name] = [d.size for d in value.shape.dim]
        ops.append({"name": node.name, "op": node.op, "inputs": list(node.input), "attrs": attrs})
        if node.op == "Const":
            array = tf.make_ndarray(node.attr["value"].tensor)
            if array.dtype.kind in "fiu" and array.size:
                constants[node.name] = array
    reader = tf.train.load_checkpoint(str(args.source / "variables/variables"))
    variables = {name: reader.get_tensor(name) for name in sorted(reader.get_variable_to_shape_map())}
    np.savez(args.output / "variables.npz", **variables)
    np.savez(args.output / "constants.npz", **constants)
    (args.output / "ops.json").write_text(json.dumps(ops, indent=2) + "\n")
    report = {"tensorflow": tf.__version__, "numpy": np.__version__, "source": identity,
              "tags": [list(m.meta_info_def.tags) for m in saved.meta_graphs],
              "variables": {k: list(v.shape) for k, v in variables.items()},
              "signatures": {name: {"inputs": {k: v.name for k, v in s.inputs.items()},
                                      "outputs": {k: v.name for k, v in s.outputs.items()}} for name, s in graph.signature_def.items()},
              "op_counts": {op: sum(n["op"] == op for n in ops) for op in sorted({n["op"] for n in ops})}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
