# SafeTensors to LiteRT conversion

## Pipeline

1. **Inspect** — locate `config.json`, tokenizer assets, SafeTensors shards, and
   an optional `model.safetensors.index.json`.
2. **Fingerprint** — hash configuration and file metadata; optionally hash every
   weight shard for release builds.
3. **Resolve adapter** — match the source `model_type` and declared architecture
   to a reviewed adapter.
4. **Load source** — load tensors without pickle and verify names, dtypes, and
   shapes against adapter expectations.
5. **Transform** — transpose/fuse/split weights and construct the supported
   inference graph. This is architecture-specific.
6. **Export** — create a SavedModel or supported intermediate representation
   with explicit prefill/decode signatures and KV-cache shapes.
7. **Convert** — invoke the LiteRT converter with a declared operator policy.
8. **Quantize** — apply a recipe supported by the pinned LiteRT exporter.
9. **Verify** — compare logits and greedy token sequences against the source;
   record tolerances and failures.
10. **Package** — emit the model, tokenizer/config assets, provenance manifest,
    checksums, validation report, and example integration metadata.

## Why adapters are mandatory

SafeTensors is a tensor storage format, not a model graph. It does not define
attention, rotary embeddings, cache updates, sampling, tokenizer behavior, or
LiteRT signatures. A supported conversion therefore requires both the original
architecture configuration and a reviewed executable graph mapping.

## Initial compatibility targets

| Family | Import | Adapter merge | LiteRT export | Status |
| --- | --- | --- | --- | --- |
| Gemma/Gemma 2 | SafeTensors | Planned | Planned via supported Google stack | Research |
| Llama 2/3-style | SafeTensors | Planned | Architecture adapter required | Research |
| Phi-3-style | SafeTensors | Planned | Architecture adapter required | Research |
| Arbitrary Transformers | Inspect only | No guarantee | Unsupported | Explicitly unsupported |

The table is a roadmap, not a compatibility claim. A family becomes supported
only after reference-logit, generation, quantization, and device tests pass.

## Quantization policies

- `none`: unquantized graph, used as the validation baseline.
- `dynamic_int8`: dynamic int8 weights with fp32 activations.
- `weight_only_int8`: int8 weights with floating-point computation.
- `dynamic_int4`: experimental dynamic blockwise int4.
- `weight_only_int4`: experimental weight-only blockwise int4.
- `static_int8`: reserved for the separate experimental calibration workflow;
  the standard exporter refuses it.
- `int4`: backend/architecture-specific and never assumed available.

## Artifact manifest

Every output records the source fingerprint, source license, adapter version,
converter/tool versions, signatures, quantization policy, validation corpus,
tolerances, target runtimes, and checksums.
