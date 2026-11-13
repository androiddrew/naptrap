"""Command-line interface: ``naptrap detect AUDIO.wav "keyword"`` and ``naptrap listen "keyword"``."""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
from pathlib import Path

from . import __version__
from .audio import load_wav
from .features import SAMPLE_RATE
from .registry import available_models, load_model
from .stream import Listener


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="naptrap", description="Text-defined keyword detection for short clips and audio streams.")
    parser.add_argument("--version", action="version", version=f"naptrap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="score one 0.5-2 s mono 16 kHz 16-bit WAV clip against a keyword")
    detect.add_argument("audio", type=Path)
    detect.add_argument("keyword")
    _model_options(detect)
    if "zp-kws" in available_models():
        detect.add_argument("--enroll", type=Path, metavar="SPEAKER.wav",
                            help="0.5-2 s clip of the enrolled speaker (experimental zp-kws model only)")
    listen = commands.add_parser(
        "listen", help="watch a stream for keywords, printing one JSON line per detection",
        description="Reads raw 16-bit little-endian mono 16 kHz PCM from standard input, or a WAV file given "
                    "with --input, and prints one JSON line for each keyword heard.")
    listen.add_argument("keywords", nargs="+", metavar="keyword")
    listen.add_argument("--input", type=Path, metavar="AUDIO.wav", help="read this WAV file instead of standard input")
    listen.add_argument("--threshold", type=float, help="score needed (default: the calibrated threshold for --far)")
    listen.add_argument("--window", type=float, default=1.5, help="seconds scored at a time, 0.5-2 (default 1.5)")
    listen.add_argument("--hop", type=float, default=0.25, help="seconds between scores (default 0.25)")
    listen.add_argument("--patience", type=int, default=1, help="windows in a row needed (default 1)")
    listen.add_argument("--refractory", type=float, default=2.0,
                        help="seconds before the same keyword is reported again (default 2)")
    _model_options(listen)
    args = parser.parse_args(argv)
    if args.command == "listen":
        return _listen(parser, args)
    try:
        model = load_model(args.model, model_dir=args.model_dir, google_dir=args.google_dir, g2p_dir=args.g2p_dir)
        options = {}
        if getattr(args, "enroll", None) is not None:
            if not hasattr(model, "enroll"):
                raise ValueError(f"--enroll is not supported by the {args.model} model")
            options["enrollment"] = model.enroll(load_wav(args.enroll))
        result = model.detect(load_wav(args.audio), args.keyword, far=args.far, **options)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"naptrap: {error}\n")
    json.dump(result.to_dict(), sys.stdout, indent=2)
    print()
    return 0 if result.accepted else 1


def _model_options(command):
    """The options that choose a model and its operating point, shared by every command."""
    command.add_argument("--model", choices=available_models(), default="phonmatchnet")
    command.add_argument("--far", type=float, default=0.01, help="calibrated operating point (default 0.01)")
    command.add_argument("--model-dir", type=Path, help="local model directory (default: download)")
    command.add_argument("--google-dir", type=Path, help="local Google speech embedding directory")
    command.add_argument("--g2p-dir", type=Path, help="local G2P decoder directory")


def _listen(parser, args):
    """Feed a stream to a Listener. Returns 0 if any keyword was heard, 1 if none was."""
    try:
        if args.input is None and sys.stdin.isatty():
            raise ValueError("no audio: pipe raw 16-bit mono 16 kHz PCM in, or give a WAV file with --input")
        model = load_model(args.model, model_dir=args.model_dir, google_dir=args.google_dir, g2p_dir=args.g2p_dir)
        listener = Listener(model, args.keywords, threshold=args.threshold, far=args.far, window=args.window,
                            hop=args.hop, patience=args.patience, refractory=args.refractory)
        if args.input is not None:
            audio = load_wav(args.input)
            chunks = (audio[start:start + SAMPLE_RATE] for start in range(0, audio.numel(), SAMPLE_RATE))
            feed = listener.feed
        else:
            chunks, feed = _pcm_chunks(sys.stdin.buffer), listener.feed_pcm16
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"naptrap: {error}\n")
    heard = False
    try:
        for chunk in chunks:
            for detection in feed(chunk):
                heard = True
                print(json.dumps(detection.to_dict()), flush=True)
    except KeyboardInterrupt:
        pass
    except OSError as error:
        if not isinstance(error, BrokenPipeError) and error.errno != errno.EINVAL:
            parser.exit(2, f"naptrap: {error}\n")
        # The reader of our output has gone (EINVAL is how Windows reports it): stop quietly, and give
        # Python somewhere to flush what is left when it exits.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except (ImportError, ValueError, RuntimeError) as error:
        parser.exit(2, f"naptrap: {error}\n")
    if not listener.scores:
        print(f"naptrap: the audio ended before one {listener.window:g} s window had been heard; nothing was scored",
              file=sys.stderr)
    return 0 if heard else 1


def _pcm_chunks(stream, samples=SAMPLE_RATE // 10):
    """Whole 16-bit samples from a byte stream, a tenth of a second at a time."""
    held = b""
    while True:
        data = stream.read(2 * samples)
        if not data:
            return
        data = held + data
        held = data[len(data) - len(data) % 2:]
        if len(data) > 1:
            yield data[:len(data) - len(data) % 2]
