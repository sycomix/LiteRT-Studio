# Ubuntu 24.04 WSL verification

Verified on 2026-07-25 using Ubuntu 24.04 under WSL2.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| LiteRT Torch | 0.9.1 |
| LiteRT-LM | 0.14.0 |
| PyTorch | 2.12.1 |
| Transformers | 4.57.6 |
| Export backend | CPU |
| Runtime backend | CPU/XNNPACK |

Transformers is constrained below version 5. LiteRT Torch 0.9.1 otherwise
installs the newest release, and Transformers 5.14.1 introduced a cache abstract
method that the exporter did not implement.

## Export results

- Source: merged LoRA version of the pinned tiny random Gemma fixture
- Tensor audit: 11 of 11 expected tensors, no missing/unexpected entries
- Signatures: `prefill_64` and `decode`
- Cache length: 128

| Variant | Recipe | Package size | SHA-256 | Export time |
| --- | --- | ---: | --- | ---: |
| Float baseline | explicit empty recipe | 37,123,508 bytes | `c718031e2b395e90554e8488cb703816de4fa669d32fdd18313795db61723f0c` | ~9 s |
| Dynamic int8 | `dynamic_wi8_afp32` | 18,687,184 bytes | `77bcd8fda5234534862b70dc6028c9a2186a789fd0196c73454bcdb18f798dd0` | ~9 s |
| Weight-only int8 | `weight_only_wi8_afp32` | 18,693,472 bytes | `b602537c1d04668d16c4e3c81932e710f280667b7198fe252615ffdcad98ad96` | ~20 s |
| Dynamic int4 | `dynamic_wi4_afp32` | 14,590,320 bytes | `413f3e6ed7e6014c0ccb9de77c9122a129dfe13188af8e9f724006126038c95b` | ~20 s |
| Weight-only int4 | `weight_only_wi4_afp32` | 14,596,608 bytes | `e7e0e5f84503cb50443242cba05cd7ba23344f40892936158e214fac938b816b` | ~20 s |

The packaged dynamic-int8 artifact is 49.66% smaller than the float baseline
(1.99x size reduction). LiteRT Torch reported the embedded TFLite graph itself
shrinking from 31.33 MiB to 13.74 MiB. An explicit empty recipe is required for
the float baseline because the exporter otherwise selects dynamic int8.

## Runtime result

LiteRT-LM loaded all five packages and delegated supported CPU operations to
XNNPACK. Both pinned prompts had exact input-token agreement with PyTorch for
every variant. Each case matched all four greedy output tokens.

| Case | Reference tokens | Float | Dynamic int8 | Result |
| --- | --- | --- | --- | --- |
| 1 | `603, 603, 603, 603` | exact | exact | Pass |
| 2 | `3454, 3454, 3454, 3454` | exact | exact | Pass |

The report records `logits_available: false` because LiteRT-LM's Python session
API exposes generated text and text-scoring values, but not the raw next-token
logit vector. Numerical-logit parity remains pending; greedy token parity is
machine verified.

## Reproduction

```bash
python -m pip install -e ".[conversion,runtime]"
python -m litert_studio.conversion.export_worker \
  --request work/wsl-export-request.json --execute

litert-studio validate-litert \
  --model /path/to/model.litertlm \
  --reference work/reference-suite.json \
  --prompts examples/reference-prompts.json \
  --output work/litert-runtime-report.json
```

The fixture is random and has no model-quality significance. This verifies
format conversion, quantization, packaging, loading, tokenization, and
deterministic decode.
