from __future__ import annotations

from pathlib import Path

SMOKE_MODEL_ID = "fxmarty/tiny-random-GemmaForCausalLM"
SMOKE_MODEL_REVISION = "ca53c1ebb8b142110b71662d702e4923e5426cb4"


def fetch_smoke_model(destination: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the 'training' extra to fetch the smoke fixture") from exc
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=SMOKE_MODEL_ID,
        revision=SMOKE_MODEL_REVISION,
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "README.md",
        ],
    )
    return destination
