#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from level1_patient_code_common import (  # noqa: E402
    deep_get,
    load_eval_model,
    load_yaml,
    require_cuda_visible_device,
    set_reproducibility,
    write_json_exclusive,
)
from train_methods import build_dataset, evaluate  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    device = require_cuda_visible_device(args.gpu)
    adapter_path = args.adapter.resolve() if args.adapter else None
    model, tokenizer = load_eval_model(cfg, adapter_path, device)
    tokenizer.padding_side = "right"

    train_dataset, eval_dataset, dropped = build_dataset(cfg, tokenizer, print)
    eval_batch_size = int(deep_get(cfg, "training.eval_batch_size", 8))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = evaluate(model, eval_dataset, eval_batch_size, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    adapter_file = (
        adapter_path / "adapter_model.safetensors" if adapter_path else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "medalpaca_utility_evaluation",
        "status": "completed",
        "label": args.label,
        "model": str(deep_get(cfg, "model.id")),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_sha256": (
            sha256_file(adapter_file)
            if adapter_file is not None and adapter_file.is_file()
            else None
        ),
        "dataset": str(deep_get(cfg, "dataset.name")),
        "medalpaca_train_samples": 7200,
        "synthetic_member_samples": int(
            deep_get(cfg, "dataset.synthetic_member_count", 0)
        ),
        "mixed_train_samples": len(train_dataset),
        "medalpaca_eval_samples": len(eval_dataset),
        "dropped_zero_response_examples": dropped,
        "max_length": int(deep_get(cfg, "dataset.max_length")),
        "eval_batch_size": eval_batch_size,
        "loss_definition": "response-only per-sequence response-token mean",
        "metrics": metrics,
        "elapsed_evaluation_sec": elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_visible": args.gpu,
        "seed": seed,
    }
    write_json_exclusive(args.output, result)
    print(f"created {args.output}: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
