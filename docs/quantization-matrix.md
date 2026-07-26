# Quantization matrix

The Studio policy name is stable project configuration. The recipe name is
the exact callable exposed by the pinned `ai-edge-quantizer` dependency used by
LiteRT Torch 0.9.1.

| Studio policy | Exporter recipe | Status | Activation |
| --- | --- | --- | --- |
| `none` | empty recipe | Baseline | fp32 |
| `dynamic_int8` | `dynamic_wi8_afp32` | Available | fp32 |
| `weight_only_int8` | `weight_only_wi8_afp32` | Available | fp32 |
| `dynamic_int4` | `dynamic_wi4_afp32` | Experimental | fp32 |
| `weight_only_int4` | `weight_only_wi4_afp32` | Experimental | fp32 |
| `static_int8` | `static_wi8_ai8` | Separate pipeline | int8 |

The standard `export_hf` worker supports the first five rows. Static int8
requires LiteRT Torch's experimental calibration, merge, and quantization
workflow and is deliberately disabled in the GUI until that pipeline is
implemented. Float16 is not exposed because it is not an
`ai-edge-quantizer` recipe accepted by the pinned `export_hf` integration.

Every model and recipe still requires reference parity and target-device
validation. Recipe availability is not a deployment-quality claim.

## Verified tiny Gemma matrix

All four quantized packages loaded with LiteRT-LM 0.14.0 on CPU/XNNPACK and
matched both four-token PyTorch reference cases exactly.

| Policy | Package bytes | Reduction from baseline | Token parity |
| --- | ---: | ---: | --- |
| `none` | 37,123,508 | — | Pass |
| `dynamic_int8` | 18,687,184 | 49.66% | Pass |
| `weight_only_int8` | 18,693,472 | 49.65% | Pass |
| `dynamic_int4` | 14,590,320 | 60.70% | Pass |
| `weight_only_int4` | 14,596,608 | 60.68% | Pass |

The fixture has random weights and cannot establish model quality. These
results validate recipe translation, export, packaging, loading, tokenization,
and deterministic decoding.
