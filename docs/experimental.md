# Experimental models

Unreleased model families ship in the source tree but stay hidden unless the
`NAPTRAP_EXPERIMENTAL` environment variable names them (comma-separated).
They are not part of any PyPI release, and their APIs and weights may change.

## zp-kws

Personalized keyword detection after ZP-KWS (Hu et al., arXiv:2606.20106),
implemented and trained independently. It adds a speaker encoder to a keyword
matcher, so a keyword only triggers for an enrolled speaker. The weights
repository, `androiddrew/naptrap-zp-kws-en`, is private for now.

```bash
export NAPTRAP_EXPERIMENTAL=zp-kws
naptrap detect clip.wav "hey computer" --model zp-kws --enroll speaker.wav
```

```python
from naptrap import load_model, load_wav

model = load_model("zp-kws")
user = model.enroll(load_wav("speaker.wav"))    # 0.5-2 s of the user; reuse it
result = model.detect(load_wav("clip.wav"), "hey computer", enrollment=user)
print(result.task, result.score, result.accepted)  # TO-KWS
```

The score is `sigmoid(keyword_logit) * sigmoid(10 * cosine - 5)`, where
`cosine` compares the speaker embeddings of the clip and the enrollment.
Without `enrollment=`, `detect()` scores the keyword alone (C-KWS) with the
keyword matcher's own threshold. `model.thresholds` lists both tasks'
operating points.

Tests for the released weights run when `NAPTRAP_TEST_ZP_KWS` points at a
local copy of the weights repository (with `NAPTRAP_TEST_MODELS` set as usual).
