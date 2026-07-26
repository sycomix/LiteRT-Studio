from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import describe_artifact


@dataclass(frozen=True)
class LogitValue:
    token_id: int
    value: float


@dataclass(frozen=True)
class ReferenceCase:
    prompt_sha256: str
    input_ids: tuple[int, ...]
    next_token_top_k: tuple[LogitValue, ...]
    generated_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReferenceSuite:
    model_fingerprint: str
    adapter_sha256: str | None
    top_k: int
    max_new_tokens: int
    cases: tuple[ReferenceCase, ...]
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def load_prompts(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Invalid reference prompt file: {path}") from exc
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) and prompt for prompt in prompts)
    ):
        raise ConfigurationError("Reference prompt file needs a non-empty 'prompts' string list")
    return prompts


def capture_reference_suite(
    model_dir: Path,
    prompts_path: Path,
    output: Path,
    *,
    adapter_dir: Path | None = None,
    top_k: int = 8,
    max_new_tokens: int = 8,
) -> ReferenceSuite:
    if top_k <= 0 or max_new_tokens <= 0:
        raise ConfigurationError("top_k and max_new_tokens must be positive")
    os.environ.setdefault("USE_TF", "0")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(prompts_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    adapter_hash: str | None = None
    if adapter_dir is not None:
        from peft import PeftModel  # type: ignore[import-not-found]

        model = PeftModel.from_pretrained(model, adapter_dir)
        weights = adapter_dir / "adapter_model.safetensors"
        adapter_hash = describe_artifact(weights).sha256
    model.eval()

    cases: list[ReferenceCase] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        with torch.inference_mode():
            logits = model(**encoded).logits[0, -1].float()
            values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]))
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_ids = tuple(int(value) for value in encoded["input_ids"][0].tolist())
        full_generated = tuple(int(value) for value in generated[0].tolist())
        cases.append(
            ReferenceCase(
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                input_ids=input_ids,
                next_token_top_k=tuple(
                    LogitValue(token_id=int(token_id), value=float(value))
                    for token_id, value in zip(
                        indices.tolist(),
                        values.tolist(),
                        strict=True,
                    )
                ),
                generated_ids=full_generated[len(input_ids) :],
            )
        )
    suite = ReferenceSuite(
        model_fingerprint=inspect_model_directory(model_dir).fingerprint,
        adapter_sha256=adapter_hash,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        cases=tuple(cases),
    )
    suite.write(output)
    return suite
