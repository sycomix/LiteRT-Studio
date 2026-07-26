# Reference backend research

Research checked on 2026-07-25 against the official Google AI Edge projects.

## Decision

The first conversion adapter will target the Gemma family through the LiteRT
Torch Generative API and package for LiteRT-LM (`.litertlm`). It is registered
at the `research` capability level until a concrete tensor mapping and parity
suite pass.

The project will preserve a separate `.tflite` target for classic models. LLM
plans must not assume the classic TensorFlow SavedModel converter is the
preferred current route.

## Evidence

- [LiteRT](https://github.com/google-ai-edge/litert) distinguishes classic
  PyTorch conversion from the LLM path and directs LLMs to the Generative Torch
  API and LiteRT-LM.
- [LiteRT Torch](https://github.com/google-ai-edge/litert-torch) describes its
  Generative API as mobile-oriented PyTorch transformer authoring and
  conversion for LiteRT-LM.
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) defines `.litertlm`
  as its model format and documents Gemma 3 models in its supported-model
  surface.

## Consequences

- Conversion workers initially target Linux because LiteRT Torch documents
  Linux as its conversion environment.
- Importing SafeTensors still requires architecture-specific re-authoring and
  weight mapping.
- Quantization choices are adapter capabilities, not global promises.
- A `.litertlm` package may contain multiple runtime components; it is not
  treated as merely a renamed flatbuffer.
