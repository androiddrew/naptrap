# Google speech-embedding dependency

## Exact artifact and publisher evidence

- Original identity: `https://tfhub.dev/google/speech_embedding/1`.
- Publisher: Google.
- Current [official model page](https://www.kaggle.com/models/google/speech-embedding).
- Exact variation: **TensorFlow1 / speech-embedding / version 1**.
- [Official API metadata](https://www.kaggle.com/api/v1/models/list?search=speech_embedding&pageSize=5)
  identifies `licenseName: "Apache 2.0"` for that variation. Rechecked October 11, 2026.
- [Publisher interface documentation](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/speech_embedding/1.md).
- [Paper: Training Keyword Spotters with Limited and Synthesized Speech Data](https://arxiv.org/abs/2002.01322).
- Acquired archive SHA-256:
  `0f626c42009c1eb916401b7b2800229a211e43100115bc591b9afbf1b60da27f`.

This is a separate pretrained speech encoder. Its output is a sequence of
96-dimensional learned embeddings, not 96 mel-frequency bins and not a
transcript. Internally, the published encoder computes its own 32-bin log-mels
and passes them through five convolutional blocks. Our keyword model's
independent 40-bin mel branch is a different input stream.

## Licensing

The **model-instance** license declaration is Apache-2.0; this is not an
inference from TensorFlow's software license alone. Apache-2.0 is permissive
and allows commercial use, modification and redistribution.

When redistributing covered material, comply with
[Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0): include the
license, retain applicable notices, reproduce relevant NOTICE content when
supplied, and prominently identify modified files. A converted model should
be identified as a conversion and retain the upstream model/version records.
Apache-2.0 does not require all of your independent application code to be
released under Apache-2.0. Trademark/endorsement permission is separate.

Naptrap's source wheel does not bundle this encoder's weights. The converted
checkpoint distribution includes the Apache-2.0 license and a NOTICE identifying
the original model and conversion. The library's own license choice and the
notices for other components remain separate.

## Input/output contract used by our checkpoints

The native Google module accepts mono 16 kHz, unit-scaled FP32 audio, shaped
`[batch,samples]`, and emits `[batch,frames,1,96]`. Without our crop adapter,
its first vector needs 12,400 samples (775 ms), followed by a new vector every
1,280 samples (80 ms).

Our frozen training/inference crop policy is:

1. Process the exact query crop at 16 kHz.
2. Prepend **11,120 zero samples**.
3. Apply the pinned Google model in inference mode, without destructive tail trimming.
4. Return **`floor(original_samples / 1280)`** valid 96D vectors.

This yields one vector per 80 ms of the original crop. Whole-recording feature
slicing does not preserve the same crop context. Any independent implementation
or conversion must reproduce this contract and pass retained numerical fixtures.

## Qualified PyTorch conversion

The acquired artifact is the legacy TensorFlow Hub / TensorFlow1 format. The
original source extractor used TensorFlow 2.15.1. Conversion/reference tooling
uses a separate TensorFlow 2.20 environment, outside Naptrap's dependencies.

`naptrap.google.GoogleSpeechEmbedding` now implements the encoder using **only
PyTorch**. The port preserves the 329,448 learned parameters, 96 source
weight/statistic tensors, original Hann window and mel matrix. Kernels are
transposed from HWIO to OIHW; the initial zero-mean kernel is materialized using
the original TensorFlow subtraction. No fine-tuning or retraining occurred.

The actual saved graph includes an initial zero-mean3×3 convolution, 19 inference
BatchNorm stages, clipped leaky activations, time/frequency pooling and a final
linear convolution producing96D vectors. The Google frontend uses **magnitude**
spectra and `log(mel+0.001)`, unlike the independent40-mel keyword branch.

### Numerical evidence

- 32 metadata-selected dev-select speech clips plus 13 deterministic stress
  fixtures cover silence, awkward lengths, noise, pure tones and impulses.
- All45 CNN comparisons given identical TensorFlow log-mels pass1e-4 absolute/
  relative tolerances. All non-tonal end-to-end comparisons retain the same gate.
- Identical windowed samples produce a maximum1.19e-6 FP32 FFT difference in a
  pure-tone diagnostic. Logarithms amplify tiny spectral-bin differences: maximum
  end-to-end feature error was approximately0.00131 on tones, or2.28e-5 relative
  to the fixture's maximum reference feature magnitude. A separate registered
  tone bound is5e-5 on that global-relative metric. The initial stricter per-
  coordinate tone failure is retained.
- Real-speech differences versus the original-machine cached features are at
  most7.25e-5. Each tested runtime checks231 downstream PhonMatchNet cases across
  all three model seeds, with both original thresholds. Maximum probability
  difference across the runtime matrix: **2.45e-6**; all decisions match.
- Python3.8/Torch1.13.1, Python3.10/Torch2.0.1, Python3.12/Torch2.10 and
  Python3.14/Torch2.10 pass. The inference processes import no TensorFlow.

The port is numerically qualified, not bit-exact across FFT backends. Evidence
is retained in the project's `runs/naptrap-google-reference-v1/`,
`runs/naptrap-google-qualification-v1/` (failed original tone gate),
`runs/naptrap-google-qualification-v2/`, and `runs/naptrap-compatibility-v1/`.
Reference tooling lives under `tools/`; trained matcher weights and frozen
benchmark results were not changed.

The port generates Google features from new audio. Text-to-G2P feature generation
is still a separate input provider, and is not replaced by this audio encoder.

Google reports YouTube pretraining and no Speech Commands dataset training.
That is provenance, not proof that every evaluation word/speaker was absent
from all upstream pretraining.
