# Compatibility

naptrap requires Python 3.9+, PyTorch 1.13+ and safetensors 0.4+. There are
no upper version pins. The library uses eager PyTorch modules only (no
TorchScript) and standard convolution, GRU and FFT operations.

The full test suite, including the released-model checks, passes on CPU with:

| Python | PyTorch | safetensors |
|---|---|---|
| 3.9 | 1.13.1 | 0.7 |
| 3.12 | 2.14.1 | 0.8 |
| 3.13 | 2.14.1 | 0.8 |

Any Python version with a PyTorch wheel for your platform should work.
Inference runs on the CPU. Scoring a 1.5 s clip takes about 7 ms using two
threads of a desktop CPU; converting a keyword takes about 2 ms.

NumPy is not required. `load_wav()` uses the standard library, and NumPy
arrays are accepted as inputs if you already have them.

The test extra installs NumPy, which safetensors needs to write the test
fixtures. With PyTorch < 2.3, use NumPy 1.x to avoid a harmless NumPy-ABI
warning.
