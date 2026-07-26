# Training and fine-tuning pipeline

LiteRT is an inference runtime; fine-tuning occurs in a training framework and
the resulting model is subsequently converted.

## Workflow

1. Inspect the base model and license.
2. Load JSONL, text, or a configured dataset provider.
3. Map records to a named prompt/chat template.
4. Validate a sample and report token-length statistics.
5. Select LoRA, QLoRA, or full fine-tuning.
6. Estimate memory and choose a local/remote runner.
7. Train with checkpointing, metric streaming, and resumable state.
8. Evaluate against a pinned suite.
9. Save adapters or merged SafeTensors.
10. hand the artifact to the conversion pipeline.

## First backend

Transformers + PEFT is the initial integration target. A generated training
launch must pin package versions, random seeds, source revisions, data
fingerprints, precision, effective batch size, and adapter targets.

The LoRA worker is now executable and verified against the pinned tiny Gemma
smoke fixture. See [Transformers/PEFT Backend](training-backend.md) for its
current capability boundary and opt-in commands.

## Dataset contract

The first local format is JSONL with either:

```json
{"text": "A fully formatted training example"}
```

or:

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
```

Dataset previews must redact configured fields and must never be included in
telemetry.

## Guardrails

- Refuse an empty dataset and malformed JSONL.
- Warn when no evaluation split or holdout strategy exists.
- Require explicit confirmation before overwriting checkpoints.
- Record model and dataset license metadata.
- Keep training and conversion environments independently reproducible.
