# Transformers/PEFT training backend

## Status

The LoRA path completed a two-step CPU smoke run on 2026-07-25 using:

- `fxmarty/tiny-random-GemmaForCausalLM`, revision
  `ca53c1ebb8b142110b71662d702e4923e5426cb4`;
- PyTorch 2.10.0;
- Transformers 4.57.6;
- PEFT 0.19.1;
- Accelerate 1.13.0;
- SafeTensors 0.7.0.

The fixture is public, ungated, MIT-licensed, uses SafeTensors, and contains
approximately 8.19 million random parameters. It tests mechanics, not model
quality.

The evaluation-enabled path was reverified under Ubuntu 24.04 WSL with
PyTorch 2.12.1, Transformers 4.57.6, PEFT 0.19.1, and Accelerate 1.14.0. A
seeded 25% split produced three training records and one evaluation record.
The two-step run completed and persisted final evaluation loss and throughput
metrics in the manifest.

## Outputs

The worker creates:

- resumable `checkpoint-*` directories;
- `metrics.jsonl` with step-level Trainer logs;
- an `adapter/` directory saved with SafeTensors;
- tokenizer/configuration assets;
- `manifest.json` with request, model, and dataset fingerprints, tool versions,
  deterministic split provenance, train/evaluation token statistics, metrics,
  file sizes, and SHA-256 hashes.

## Evaluation behavior

Set either `validation_split` or `eval_dataset`, but not both. A validation
split is shuffled deterministically from the configured training seed, always
leaves at least one training and one evaluation record, and is connected to
the Trainer evaluation loop. A separate evaluation JSONL is independently
validated and fingerprinted. Final evaluation metrics are stored in the
artifact manifest.

## LoRA behavior

`target_modules: "auto"` maps to PEFT's `all-linear` selection. An explicit
list can be used for reviewed architecture recipes. The model loads with
`trust_remote_code=False`, cache use is disabled during training, and the
worker does not enable telemetry.

## QLoRA behavior

QLoRA is a separate path using a 4-bit NF4 `BitsAndBytesConfig`, double
quantization, and `prepare_model_for_kbit_training`. It requires the `qlora`
extra and CUDA in the initial backend. Preflight reports missing
`bitsandbytes`; execution refuses CPU QLoRA rather than silently falling back
to ordinary LoRA.

## Known limitations

- Dataset packing and completion-only loss masks are not implemented.
- The worker currently supports local model directories, not direct registry
  identifiers.
- Hardware probing and memory estimation remain separate future services.
- Adapter merging is not yet enabled.
