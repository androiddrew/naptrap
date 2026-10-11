# Dependency and attribution scope

- **PhonMatchNet:** the independently trained model uses the licensed NCSOFT
  PhonMatchNet foundation (BSD-3-Clause). The original notice is reproduced in
  `licenses/PhonMatchNet-BSD-3-Clause.txt`. The training checkout is not imported
  by this library. Exact model/asset provenance and notices accompany the
  separately distributed model bundle.
- **Google speech_embedding/1:** publisher-declared Apache-2.0 model-instance
  license. It is a required feature source for the trained keyword models, but
  its weights and TensorFlow runtime are not bundled in this source package.
  `naptrap.google` independently implements its inference graph using PyTorch;
  the converted weight distribution preserves the upstream Apache-2.0 license
  and conversion notice. See `docs/google-embedding.md`.
- **G2P:** the trained models expect the licensed neural decoder's 256D hidden
  states. The current source project's modified G2P adapter and weights are
  Apache-2.0, with additional resource-specific notices. Naptrap currently
  accepts the resulting features; it does not vendor that implementation.
- **PyTorch / NumPy:** separately installed runtime dependencies; their native
  libraries are not vendored here.

The trained model bundle contains exact mel/window buffers. Reusing those
artifacts or adding a new wrapper does not remove applicable upstream notices.
The final license for the new Naptrap code and our original learned weights is
still an explicit project choice; dependency licenses do not automatically
assign that choice.
