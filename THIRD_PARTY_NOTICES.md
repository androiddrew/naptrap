# Third-party notices

naptrap's own code is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`).
The package bundles **no model weights**. Weights are downloaded separately
from Hugging Face, and each repository carries its own license and notices.

| Component | Used for | License | Where |
|---|---|---|---|
| PhonMatchNet architecture and training code, © NCSOFT Corporation | `naptrap/models/network.py` follows its network design | BSD-3-Clause | `licenses/PhonMatchNet-BSD-3-Clause.txt`, <https://github.com/ncsoft/PhonMatchNet> |
| PhonMatchNet weights, trained independently by Drew Bednar | default keyword model | Apache-2.0 | <https://huggingface.co/androiddrew/naptrap-phonmatchnet-en> |
| Google speech_embedding/1 | frozen audio features (`naptrap/google.py`) | Apache-2.0 | <https://huggingface.co/androiddrew/google-speech-embedding-pytorch> |
| g2p_en 2.1.0, Kyubyong Park & Jongseok Kim | frozen keyword text features (`naptrap/text.py`) | Apache-2.0 | <https://huggingface.co/androiddrew/g2p-en-decoder-pytorch> |
| LibriSpeech (Panayotov et al.) | training data for the PhonMatchNet weights | CC BY 4.0 | <https://www.openslr.org/12> |
| PyTorch, safetensors, huggingface_hub (optional) | runtime dependencies, installed separately | BSD-3-Clause / Apache-2.0 / Apache-2.0 | PyPI |

naptrap and these weights are not affiliated with or endorsed by NCSOFT, the
PhonMatchNet authors, Google or the g2p_en authors.
