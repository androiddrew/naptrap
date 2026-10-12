# naptrap

**Local, text-defined wake word detection in PyTorch.** Type a keyword, give
naptrap a short audio clip, and it tells you whether the keyword was spoken.
New wake words need no recordings or training.

```python
from naptrap import load_model, load_wav

model = load_model("phonmatchnet")             # downloads the model on first use
result = model.detect(load_wav("clip.wav"), "hey computer")

print(result.score)     # keyword-match probability, 0-1
print(result.accepted)  # score >= the model's calibrated threshold
```

The first model is an independently trained
[PhonMatchNet](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en)
(Lee & Cho, Interspeech 2023) for English. Inference runs on the CPU with
**PyTorch as the only heavy dependency**. There is no TensorFlow, NumPy,
torchaudio or NLTK, and no cloud API.

## Install

```bash
pip install "naptrap[hub]"
```

Python 3.9+ and PyTorch 1.13+ are required. If you want a CPU-only or
CUDA-specific PyTorch build, install it first; see
[pytorch.org](https://pytorch.org/get-started/locally/). The `hub` extra adds
`huggingface_hub` for automatic model downloads. Without it, `pip install
naptrap` works with model directories you download yourself.

## Models

`load_model("phonmatchnet")` fetches three Hugging Face repositories (about
6 MB in total), pinned to revision `v1`, and caches them locally:

| Repository | Role | License |
|---|---|---|
| [`androiddrew/naptrap-phonmatchnet-en`](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en) | keyword matcher | Apache-2.0 |
| [`androiddrew/google-speech-embedding-pytorch`](https://huggingface.co/androiddrew/google-speech-embedding-pytorch) | frozen audio encoder | Apache-2.0 |
| [`androiddrew/g2p-en-decoder-pytorch`](https://huggingface.co/androiddrew/g2p-en-decoder-pytorch) | frozen keyword text encoder | Apache-2.0 |

To work offline, download them once and pass the directories:

```bash
hf download androiddrew/naptrap-phonmatchnet-en --revision v1 --local-dir models/phonmatchnet
hf download androiddrew/google-speech-embedding-pytorch --revision v1 --local-dir models/google
hf download androiddrew/g2p-en-decoder-pytorch --revision v1 --local-dir models/g2p
```

```python
model = load_model("phonmatchnet", model_dir="models/phonmatchnet",
                   google_dir="models/google", g2p_dir="models/g2p")
```

Every weight file is checked against the SHA-256 recorded in its `config.json`.
The keyword model also pins the exact encoders it was trained with.

## Usage

### Audio

`detect()` scores **one mono 16 kHz clip of 0.5–2 seconds**, given as a 1-D
float32 tensor or array with values in [-1, 1]. `load_wav()` reads 16-bit PCM
WAV files using only the standard library. Convert other files first:

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 -sample_fmt s16 clip.wav
```

naptrap does not resample, trim or rescale audio for you.

### Reusing a keyword

Converting the keyword text is cheap, but if you score many clips against the
same keyword you can do it once:

```python
wake = model.keyword("hey computer")
for clip in clips:
    if model.detect(clip, wake).accepted:
        ...
```

Keywords are English words; accents and punctuation are removed. Spell out
numbers ("route sixty six").

### Operating points and thresholds

```python
model.thresholds                                # {0.01: 0.7149..., 0.1: 0.0025...}
model.detect(clip, "hey computer", far=0.1)     # more sensitive
```

Each threshold was fitted on held-out development speakers to target a
per-clip false-accept rate (FAR) of 1% or 10%. **These targets do not hold on
other audio.** On phonetically confusable phrases, the 1% threshold accepted
31% of negatives. For a real product, record some positive and negative clips
in your setting and choose your own threshold on `result.score`. See the
[model card](https://huggingface.co/androiddrew/naptrap-phonmatchnet-en) for
the evaluation.

### Command line

```bash
naptrap detect clip.wav "hey computer"
naptrap detect clip.wav "hey computer" --far 0.1 --model-dir models/phonmatchnet \
    --google-dir models/google --g2p-dir models/g2p
```

The command prints the prediction as JSON. It exits with 0 if the keyword was
accepted, 1 if it was rejected and 2 on error. `python -m naptrap` works the
same way.

### Precomputed features

`model.predict({"keyword_mel": ..., "google": ..., "text": ...})` scores
features you computed yourself: `[T, 40]`, `[T/8, 96]` and `[P, 256]` float32.
The encoders are exposed as `naptrap.GoogleSpeechEmbedding` and
`naptrap.G2PTextEncoder`.

## Limitations

- **Single clips, not a streaming listener (yet).** To watch a microphone you
  must window the audio yourself, e.g. 1.5 s windows every 0.25 s, and add
  your own debouncing.
- **English only**, trained on read audiobook speech (LibriSpeech). Expect
  lower accuracy with noise, far-field microphones, accents or children.
- **No speaker verification.** Anyone saying the keyword triggers it.
- Similar-sounding phrases are often confused.
- CPU inference is the supported path; other devices have not been validated.
- Not for security or safety-critical use.

## Adding model families

Backends implement `keyword()`, `detect()` and `predict()`, and return a
`Prediction`. Register a loader under a new name:

```python
from naptrap import register_model, load_model

register_model("my-model", MyModel.from_pretrained)
model = load_model("my-model", model_dir="...")
```

## Development

```bash
pip install -e ".[test]"
pytest
# Also run the real-model tests against downloaded model directories:
NAPTRAP_TEST_MODELS=path/to/models pytest
```

`NAPTRAP_TEST_MODELS` must point at a directory holding the three model
repositories in subdirectories named `naptrap-phonmatchnet-en`,
`google-speech-embedding-pytorch` and `g2p-en-decoder-pytorch`.

## License and attribution

naptrap is licensed under the [Apache License 2.0](LICENSE). The network
follows the PhonMatchNet design and NCSOFT's BSD-3-Clause reference code (see
[NOTICE](NOTICE) and [licenses/](licenses/)). The PhonMatchNet weights were
trained independently. **naptrap is not affiliated with or endorsed by NCSOFT
or the PhonMatchNet authors.** See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for every component.

If naptrap's PhonMatchNet model is useful in your research, please cite:

```bibtex
@inproceedings{lee23phonmatchnet,
  title     = {PhonMatchNet: Phoneme-Guided Zero-Shot Keyword Spotting for User-Defined Keywords},
  author    = {Lee, Yong-Hyeok and Cho, Namhyun},
  booktitle = {Proc. INTERSPEECH 2023},
  pages     = {3964--3968},
  year      = {2023}
}
```
