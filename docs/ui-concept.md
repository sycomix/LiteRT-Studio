# Studio UI concept

## Navigation

The desktop-shaped web app has six primary workspaces:

1. **Projects** — recent projects, storage, source/target summary.
2. **Models** — import, inspect tensors/config, compatibility report.
3. **Data** — schema mapping, preview, formatting, token statistics.
4. **Train** — recipe builder, memory estimate, live metrics, checkpoints.
5. **Convert** — adapter, signatures, quantization, validation matrix.
6. **Artifacts** — manifests, benchmarks, checksums, packaging/export.

## Main conversion flow

```text
┌ Source model ─────┐  ┌ Compatibility ───┐  ┌ Export target ────┐
│ local / registry  │→ │ adapter + issues │→ │ signatures + ops  │
└───────────────────┘  └───────────────────┘  └───────────────────┘
          ↓                      ↓                       ↓
┌ Tensor inspector ┐  ┌ Quantization ─────┐  ┌ Validate/package ┐
│ names/dtypes/size │  │ policy + dataset │→ │ logits/device    │
└───────────────────┘  └───────────────────┘  └──────────────────┘
```

## Interaction rules

- Show a readable plan before every run.
- Label unsupported combinations at selection time.
- Keep warnings adjacent to the setting that causes them.
- Separate measured results from estimates.
- Make logs searchable but keep the default view task-oriented.
- Allow every UI-created plan to be exported as JSON and replayed by CLI.

## Component map

`ProjectShell`, `SourcePicker`, `ModelInspector`, `CompatibilityCard`,
`DatasetMapper`, `RecipeBuilder`, `ResourceEstimate`, `PipelineGraph`,
`JobConsole`, `MetricChart`, `QuantizationPicker`, `ValidationMatrix`, and
`ArtifactManifest`.

See `apps/studio-web/README.md` for the proposed client implementation.
