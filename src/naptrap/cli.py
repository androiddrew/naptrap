"""Feature-input command-line interface for the current PhonMatchNet backend."""

import argparse
import json
from pathlib import Path

from .registry import available_models, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=available_models(), default="phonmatchnet")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="NPZ containing keyword_mel, google and text")
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--far", type=float, choices=(.01, .1), default=.01)
    args = parser.parse_args()
    try:
        import numpy as np
    except ImportError:
        parser.exit(2, "naptrap: NPZ input requires the optional 'examples' extra: pip install 'naptrap[examples]'\n")
    try:
        model = load_model(args.model, bundle=args.bundle, seed=args.seed)
        with np.load(args.features, allow_pickle=False) as features:
            result = model.predict(features, far=args.far)
        print(json.dumps(result.to_dict(), indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"naptrap: {error}\n")
