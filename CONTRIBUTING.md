# Contributing

Start with an issue describing the model family, source checkpoint format,
target runtime, and validation device. Conversion support must include:

- an explicit architecture adapter;
- a tensor-name/shape mapping test;
- numerical comparison against the source model;
- a documented quantization matrix;
- artifact provenance.

Run `python -m pytest` and `ruff check .` before submitting changes. Never add
model weights, private datasets, access tokens, or generated artifacts to Git.
