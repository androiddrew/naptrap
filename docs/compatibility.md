# Compatibility and dependency policy

Naptrap's runtime uses **PyTorch only**. NumPy is an optional extra for NPZ files
and examples; test/conversion tools have separately scoped dependencies.
There is no TensorFlow, torchaudio, ONNX Runtime, SciPy or NLTK runtime import.

## Supported floor and measured combinations

Metadata permits Python **3.8+** and PyTorch **1.13+**, without arbitrary upper
pins. The supported API uses eager modules, stable convolution/GRU/FFT operations,
and state dictionaries loaded with `weights_only=True`. It avoids relying on
TorchScript loading, which PyTorch warns is unsupported on Python3.14+.

| Python | PyTorch | CPU model tests | Google/downstream fixture qualification |
|---|---|---|---|
|3.8.20|1.13.1|Passed|45 encoder fixtures,231 downstream cases passed|
|3.10.19|2.0.1|Passed|45 encoder fixtures,231 downstream cases passed|
|3.12.3|2.10.0+cpu|Passed|45 encoder fixtures,231 downstream cases passed|
|3.14.0|2.10.0+cpu|Passed|45 encoder fixtures,231 downstream cases passed|

These are tested combinations, not every possible Cartesian pairing. A Python
version still needs a compatible PyTorch wheel for its OS/architecture. Old
PyTorch CPU support does not imply GB10 CUDA support. The initial public backend
is CPU; additional accelerators require their own numeric checks.

NumPy examples on old PyTorch should use NumPy1.x; the matrix used1.24.4 on
Python3.8 and1.26.4 on3.10/3.12. Python3.14 used NumPy2.3.4. Core calls with
`torch.Tensor` inputs do not need NumPy.

## Dependency-isolation check

`tools/check_torch_only.py` blocks imports of TensorFlow, NumPy, torchaudio,
ONNX Runtime, SciPy, NLTK and the original `zp_kws` project before importing
Naptrap. It loads the real converted Google weights and selected PhonMatchNet
weights and performs inference on engineering tensors. It passes on the oldest
tested runtime and the modern Python3.12/3.14 runtimes.

PyTorch itself may emit its standard warning when optional NumPy initialization
is blocked; inference succeeds without it. This is not a hidden NumPy dependency.

## Conversion tools

TensorFlow is only required to reproduce the one-time original-model inspection
and reference extraction. The checked source graph, parameter mappings and
numerical gates are retained in `tools/` and the parent project's versioned
qualification runs. End users load the converted checkpoint directly.

The pretrained assets are external files, excluded from the library wheel and
Git source tree. Their licenses and versions are independent of Python package
dependency versions.
