# LiteRT Studio

LiteRT Studio is an open-source, local-first workbench for taking LLMs from
SafeTensors checkpoints to fine-tuned, optimized, and deployable LiteRT
(formerly TensorFlow Lite) artifacts.

> [!IMPORTANT]
> This repository is an early implementation. Training, LiteRT export, runtime
> validation, and packaging have verified fixture-scale paths. Real model
> families remain research-grade until each model/quantization/device
> combination passes the documented validation gates.

## Why this project?

LLM tooling is usually split across training scripts, conversion notebooks,
quantization utilities, and device-specific deployment examples. LiteRT Studio
aims to put that workflow behind one reproducible project model:

1. Import a Hugging Face-style SafeTensors model.
2. Inspect architecture, tokenizer, tensor index, and compatibility.
3. Fine-tune with adapters or a supported full-training backend.
4. Merge or retain adapters.
5. Export through a supported bridge into a LiteRT model.
6. Quantize, validate, benchmark, and package for Android or edge runtimes.

The product experience is inspired by the accessibility of Unsloth Studio, but
the implementation and branding are independent and focused specifically on
LiteRT deployment.

## Current scope

- Typed, serializable conversion and training job plans
- SafeTensors directory inspection and sharded-index validation
- Versioned JSON schemas and explicit backend capability contracts
- SQLite job lifecycle/event persistence and a local runner foundation
- Content-based dataset fingerprinting without retaining preview records
- Isolated training-worker requests and side-effect-free dependency preflight
- Shell-free subprocess launching with environment allowlisting
- Numerical parity, token parity, and artifact-manifest contracts
- Explicit conversion stages with pluggable backend boundaries
- Fine-tuning plan generation for LoRA/QLoRA/full workflows
- Executable LoRA training with resumable checkpoints and structured metrics
- Complete local interface plus CLI workflows
- Architecture, UI, security, roadmap, and compatibility documentation
- Unit tests that do not download models or require a GPU

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
litert-studio init demo
litert-studio plan-convert examples/conversion.json
litert-studio inspect-gemma models/gemma-demo
litert-studio prepare-export examples/conversion.json --output runs/export-request.json
litert-studio plan-train examples/training.json
litert-studio prepare-train examples/training.json --output runs/training-request.json
python -m litert_studio.training.worker --request runs/training-request.json --preflight
python -m pytest
```

To launch the complete local interface:

```bash
python -m pip install -e ".[api,training,conversion,runtime]"
litert-studio serve --workspace .
```

Open `http://127.0.0.1:7860`. Launch Studio inside Ubuntu 24.04 WSL when using
the Linux-only LiteRT conversion and runtime actions. See
[Local Studio interface](docs/studio-gui.md).

The example model and dataset paths are placeholders. Replace them before
running validation against real local assets.

To run the pinned CPU-sized Gemma smoke fixture:

```bash
python -m pip install -e ".[training]"
litert-studio fetch-smoke-fixture
litert-studio prepare-train examples/training-smoke.json --output runs/smoke-request.json
python -m litert_studio.training.worker --request runs/smoke-request.json --execute
```

Downloading is explicit, revision-pinned, and limited to the MIT-licensed
`fxmarty/tiny-random-GemmaForCausalLM` test model. Random fixture output has no
quality value; it validates the pipeline only.

The SafeTensors-to-LiteRT bridge is prepared on Windows and executed on Linux:

```bash
litert-studio merge-adapter \
  --base-model models/gemma-demo \
  --adapter artifacts/gemma-demo-lora/adapter \
  --output artifacts/gemma-demo-merged

litert-studio inspect-gemma artifacts/gemma-demo-merged
python -m litert_studio.conversion.export_worker \
  --request runs/export-request.json --execute

litert-studio validate-litert \
  --model artifacts/model.litertlm \
  --reference runs/reference-suite.json \
  --prompts examples/reference-prompts.json \
  --output runs/litert-runtime-report.json

litert-studio package-artifact artifacts/export \
  --report runs/litert-runtime-report.json \
  --output releases/model.litertstudio

litert-studio verify-bundle releases/model.litertstudio
```

See [LiteRT Export](docs/litert-export.md) for the Linux environment gate and
current exporter settings.

The pinned tiny Gemma workflow completed this path under Ubuntu 24.04 WSL for
both float and dynamic-int8 exports. Both packages load in LiteRT-LM on CPU and
pass exact greedy-token parity; the quantized package is 49.66% smaller. See
[WSL Verification](docs/wsl-verification.md).

Weight-only int8 plus dynamic and weight-only int4 have also passed the same
fixture-scale runtime gate. The int4 packages are approximately 60.7% smaller
than the float baseline. See
[Quantization matrix](docs/quantization-matrix.md).

## Repository map

```text
src/litert_studio/
  cli.py                 Command-line entry point
  core/                  Jobs, persistence, runners, capabilities, serialization
  conversion/            SafeTensors inspection and export planning
  training/              Dataset/fine-tuning planning
  server/                Local API and responsive Studio interface
apps/studio-web/          UI architecture notes
docs/                     Architecture and product decisions
examples/                 Versioned example configurations
tests/                    Lightweight contract tests
```

## Product principles

- **Local first:** model weights and datasets stay on the user's machine unless
  they explicitly configure a remote runner.
- **Plans before execution:** every operation can be inspected and saved before
  expensive compute begins.
- **No magical conversion:** architecture support, tensor transforms, and
  quantization constraints must be visible.
- **Reproducible outputs:** every artifact includes source fingerprints,
  settings, tool versions, validation results, and licensing metadata.
- **Backend-neutral orchestration:** the Studio coordinates proven tools rather
  than hiding them inside an inseparable monolith.

## Non-goals for the first release

- Arbitrary PyTorch graph conversion
- Training directly inside the LiteRT runtime
- Silent upload of private models or datasets
- Claiming compatibility without numerical and on-device validation

See [Architecture](docs/architecture.md), [Conversion Pipeline](docs/conversion-pipeline.md),
[Training Pipeline](docs/training-pipeline.md), [UI Concept](docs/ui-concept.md),
[Roadmap](docs/roadmap.md), and
[Reference Backend Research](docs/reference-backend-research.md). Worker
isolation is documented in [Execution Model](docs/execution-model.md).

## License

Apache-2.0. Model weights, datasets, and generated artifacts retain their own
licenses; LiteRT Studio does not change those terms.
