"""Prove real inference works with optional packages blocked from importing."""

import argparse
import importlib.abc
import json
import sys


BLOCKED = {"tensorflow", "numpy", "torchaudio", "onnxruntime", "nltk", "scipy", "zp_kws"}


class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BLOCKED:
            raise ModuleNotFoundError("Dependency intentionally blocked: " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--google", required=True)
    args = parser.parse_args()
    if any(name.split(".", 1)[0] in BLOCKED for name in sys.modules):
        raise RuntimeError("Dependency isolation requires a clean interpreter")
    sys.meta_path.insert(0, BlockOptional())
    import torch
    import naptrap
    torch.set_num_threads(2)
    encoder = naptrap.GoogleSpeechEmbedding.from_directory(args.google)
    model = naptrap.load_model(bundle=args.bundle, google_encoder=encoder)
    result = model.predict_audio(torch.zeros(8000), text=torch.zeros(6, 256))
    imported = sorted({name.split(".", 1)[0] for name in sys.modules} & BLOCKED)
    if imported:
        raise RuntimeError("Unexpected inference dependencies: " + repr(imported))
    print(json.dumps({"stage": "passed", "torch": str(torch.__version__), "python": sys.version,
                      "blocked_dependencies": sorted(BLOCKED), "optional_modules_imported": imported,
                      "score": result.score, "features": "synthetic silence/text-zero engineering smoke test only"}, indent=2))


if __name__ == "__main__":
    main()
