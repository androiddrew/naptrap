"""Command-line interface: ``naptrap detect AUDIO.wav "keyword"``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .audio import load_wav
from .registry import available_models, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(prog="naptrap", description="Text-defined keyword detection for short clips.")
    parser.add_argument("--version", action="version", version=f"naptrap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="score one 0.5-2 s mono 16 kHz 16-bit WAV clip against a keyword")
    detect.add_argument("audio", type=Path)
    detect.add_argument("keyword")
    detect.add_argument("--model", choices=available_models(), default="phonmatchnet")
    detect.add_argument("--far", type=float, default=0.01, help="calibrated operating point (default 0.01)")
    detect.add_argument("--model-dir", type=Path, help="local model directory (default: download)")
    detect.add_argument("--google-dir", type=Path, help="local Google speech embedding directory")
    detect.add_argument("--g2p-dir", type=Path, help="local G2P decoder directory")
    args = parser.parse_args(argv)
    try:
        model = load_model(args.model, model_dir=args.model_dir, google_dir=args.google_dir, g2p_dir=args.g2p_dir)
        result = model.detect(load_wav(args.audio), args.keyword, far=args.far)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"naptrap: {error}\n")
    json.dump(result.to_dict(), sys.stdout, indent=2)
    print()
    return 0 if result.accepted else 1
