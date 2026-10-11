# naptrap

**Local Python inference for text-defined keyword models.** The first backend
is our independently trained **PhonMatchNet**. A small model registry leaves
room for other families, including our ZP-KWS-style system later.

This package loads the existing, qualified model bundle. It does not import
the PhonMatchNet training repository, retrain weights, or download assets at
import time. Model files and training data are kept outside the source package.

## Install

The core requires **Python 3.8+ and PyTorch 1.13+**, with no upper version pin.
**PyTorch is the only direct runtime dependency.** TensorFlow, NumPy,
torchaudio, ONNX Runtime, SciPy and NLTK are not needed by the core inference API.

```bash
python -m pip install -e ./naptrap
```

Choose an official PyTorch wheel compatible with your Python/platform. If
PyTorch is already installed, install just this package with `--no-deps`.
For the NPZ command-line examples below, install the optional NumPy extra:

```bash
python -m pip install -e './naptrap[examples]'
```

### Tested CPU combinations

| Python | PyTorch | Optional test/example NumPy |
|---|---|---|
| 3.8.20 | 1.13.1 | 1.24.4 |
| 3.10.19 | 2.0.1 | 1.26.4 |
| 3.12.3 | 2.10.0+cpu | 1.26.4 |
| 3.14.0 | 2.10.0+cpu | 2.3.4 |

All four pass the retained model and Google-encoder numerical checks. This is
not a promise that every Python/PyTorch pair has a compatible wheel. Older
PyTorch with NumPy examples should use NumPy 1.x. GB10 CUDA support requires a
newer GPU-compatible PyTorch build; this initial library backend is qualified
on CPU. Ordinary eager PyTorch avoids TorchScript's Python 3.14 compatibility
warning. See [docs/compatibility.md](docs/compatibility.md).

## Try the existing model bundle

```python
from pathlib import Path
import numpy as np
from naptrap import load_model

bundle = Path.home() / "wakeword-models/releases/python-onnx-v1"
model = load_model("phonmatchnet", bundle=bundle, seed=17)

with np.load(bundle / "examples/said-mister-seed17.npz", allow_pickle=False) as features:
    prediction = model.predict(features)

print(prediction.score)      # Keyword-match probability
print(prediction.accepted)   # Original dev-cal threshold, C-KWS task
print(prediction.to_dict())
```

Or use the command line:

```bash
naptrap --bundle ~/wakeword-models/releases/python-onnx-v1 \
  --features ~/wakeword-models/releases/python-onnx-v1/examples/said-mister-seed17.npz
```

`python -m naptrap` offers the same options. Seed 17 is our fixed convenience
default, not a winner chosen from final-test results. Seeds 29 and 43 are also
available. `far=0.01` or `far=0.1` chooses an existing development-calibrated
operating point; it does not promise that false-acceptance rate on new audio.

## Inputs in this first version

The current backend is **CPU eager PyTorch, one unpadded query clip at a time**.
It returns content-only keyword detection (C-KWS), without speaker enrollment.

| Feature key | Shape, FP32 | Meaning |
|---|---|---|
| `keyword_mel` | `[T,40]` or `[1,T,40]`, 50–200 frames | Our trainable audio branch's exact log-mel input |
| `google` | `[G,96]` or `[1,G,96]`, 6–25 frames | Frozen Google speech-embedding vectors |
| `text` | `[P,256]` or `[1,P,256]`, 1–128 positions | Frozen neural G2P decoder features for the desired keyword |

Our clip contract is **mono 16 kHz, 0.5–2 seconds**. See the Google contract
below for the distinct feature clock. Arrays must be finite float32. Extra keys
in the example NPZ (such as speaker features) are unused by this backend.

Naptrap can calculate its own mel branch from a waveform:

```python
# waveform: mono float32 samples scaled once into [-1,1]
# google_features: extracted for this exact waveform/crop
# text_features: neural G2P features for the desired keyword
prediction = model.predict_audio(
    waveform,
    google=google_features,
    text=text_features,
    sample_rate=16000,
)
```

You can now have Naptrap compute the Google features using the converted
PyTorch encoder as well:

```python
from naptrap import GoogleSpeechEmbedding, load_model

google_encoder = GoogleSpeechEmbedding.from_directory(
    "~/wakeword-models/encoders/google-speech-embedding-pytorch-v1"
)
model = load_model("phonmatchnet", bundle=bundle, google_encoder=google_encoder)
prediction = model.predict_audio(waveform, text=text_features)
```

The Google encoder runs locally with PyTorch only. You still supply the keyword's
matching **256D neural G2P features**; this is not yet a `WAV + arbitrary keyword
string` frontend. Ordinary phoneme strings or another encoder's features are
not an equivalent replacement for the trained inputs.

## What is the Google model, and does it require TensorFlow?

It is **`google/speech_embedding/1`**, a small pretrained convolutional speech
encoder. It maps audio to a sequence of **96-number vectors** describing speech
patterns. It is not a transcript-producing recognizer or a cloud API; its
downloaded weights can be run locally. Both our PhonMatchNet and ZP-KWS-style
models were trained using these frozen features.

Google's official Kaggle model-instance metadata identifies the exact
**TensorFlow1 / speech-embedding / version 1** artifact as **Apache 2.0**.
That is a permissive license allowing commercial use and redistribution,
subject to its license/notice and modification-marking conditions.

The original artifact is a legacy TensorFlow model. **A validated PyTorch port
is now implemented in `naptrap.google`**, using all 96 original weight/statistic
tensors and the exact frontend constants. The converted checkpoint is about
1.4 MB and is distributed separately from the library. TensorFlow is used only
by the offline conversion/reference tools; it is not imported for inference.

The port uses the same model mathematics and weights, with measured floating-
point differences between FFT implementations. Across the tested runtime matrix,
the maximum downstream probability difference was **2.45e-6**, with identical
decisions at the frozen thresholds. Pure-tone stress differences are documented,
not described as bit-exact conversion. See the qualification details below.

See [docs/google-embedding.md](docs/google-embedding.md) for publisher links,
the precise preprocessing contract and redistribution obligations.

## Adding another model

Backends return the common `Prediction` result and implement `predict(features,
far=...)`. Register a loader explicitly:

```python
from naptrap import register_model, load_model

# FutureZpKws.from_bundle must return a backend implementing KeywordModel.
register_model("zp-kws", FutureZpKws.from_bundle)
detector = load_model("zp-kws", bundle=another_bundle)
```

Only `phonmatchnet` is built in today. Model-specific enrollment, features and
scoring semantics belong in the corresponding backend. No future model is
silently substituted for the requested family.

## Reproducibility and tests

Model, frontend and calibration files are checked against the bundle manifest.
Calibration must match the selected seed, PhonMatchNet variant and dev-cal role.
Checksums establish bundle integrity; obtain the manifest from a trusted source.

```bash
python -m pip install -e './naptrap[test]'
python -m pytest naptrap/tests -q

# Enable real-model tests against your external candidate bundle:
NAPTRAP_TEST_BUNDLE=~/wakeword-models/releases/python-onnx-v1 \
  python -m pytest naptrap/tests -q

# Include the converted Google encoder's integration cases:
NAPTRAP_TEST_BUNDLE=~/wakeword-models/releases/python-onnx-v1 \
NAPTRAP_TEST_GOOGLE=~/wakeword-models/encoders/google-speech-embedding-pytorch-v1 \
  python -m pytest naptrap/tests -q
```

The model bundle, not this source tree, contains trained parameters and frontend
buffers. Keep it separately available when moving or installing the library.

## License scope

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model's BSD-3-Clause
foundation and the Google encoder's Apache-2.0 terms are separate from the
license choice for our new library and learned weights. That project-level
choice has not yet been made; no blanket license is assigned by this README.
