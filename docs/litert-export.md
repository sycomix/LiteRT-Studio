# Gemma to LiteRT-LM export

## Current boundary

The cross-platform preparation path is implemented and verified:

- SafeTensors headers are parsed without loading tensor payloads.
- Gemma/Gemma 2 layer names, completeness, ranks, and adapter-only checkpoints
  are audited.
- PEFT adapters can be safely merged into a full float32 SafeTensors model.
- Export settings are materialized into a deterministic request.
- PyTorch reference logits and greedy tokens can be captured before export.

Actual LiteRT Torch generative conversion is Linux-only, matching the platform
requirement published by the project. The pinned tiny Gemma baseline has now
been exported and loaded successfully under Ubuntu 24.04 WSL.

## Official API integration

The worker calls:

```python
from litert_torch.generative.export_hf import export as export_module

export_module.export(
    model=request.model,
    output_dir=request.output_dir,
    task="text_generation",
    trust_remote_code=False,
    prefill_lengths=list(request.prefill_lengths),
    cache_length=request.cache_length,
    quantization_recipe=request.quantization_recipe or "",
    externalize_embedder=request.externalize_embedder,
    use_jinja_template=request.use_jinja_template,
    bundle_litert_lm=True,
)
```

This follows the current
[LiteRT Torch](https://github.com/google-ai-edge/litert-torch) `export_hf`
implementation rather than relying on CLI shorthand. The output target is the
`.litertlm` package used by
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM).

## Workflow

```bash
litert-studio merge-adapter \
  --base-model models/gemma \
  --adapter artifacts/gemma-lora/adapter \
  --output artifacts/gemma-merged

litert-studio inspect-gemma artifacts/gemma-merged

litert-studio capture-reference \
  --model models/gemma \
  --adapter artifacts/gemma-lora/adapter \
  --prompts examples/reference-prompts.json \
  --output runs/reference-suite.json

litert-studio prepare-export examples/conversion.json \
  --output runs/export-request.json

python -m litert_studio.conversion.export_worker \
  --request runs/export-request.json --execute
```

## Validation gate

Successful serialization is not sufficient. The exported model must be run
through LiteRT-LM with the same input token IDs. Candidate logits at the
recorded top-k token IDs feed the numerical parity report, and greedy output
feeds token parity. Promotion from `research` requires both parity checks plus
artifact loading and a target-device smoke test.

The float and `dynamic_wi8_afp32` reference fixtures both pass exact
greedy-token parity on LiteRT-LM CPU. The quantized package is 49.66% smaller
than the float package. Raw-logit parity remains pending because the Python
runtime API does not expose next-token logits. See
[WSL Verification](wsl-verification.md).

## Known risks

- LiteRT Torch's Generative API is still marked alpha.
- Export can require multiple in-memory copies of model weights.
- Quantization recipe support varies by architecture and release.
- Current upstream issue reports show that an export may serialize yet fail at
  runtime, reinforcing the mandatory parity/device gate.
