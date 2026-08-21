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


def execute(args: argparse.Namespace) -> dict[str, Any]:
    level1_cfg = load_yaml(args.level1_config)
    benchmark_cfg = load_yaml(args.benchmark_config)
    seed = int(deep_get(level1_cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    device = require_cuda_visible_device(args.gpu)
    adapter_path = args.adapter_path.resolve() if args.adapter_path else None
    model, tokenizer = load_eval_model(level1_cfg, adapter_path, device)
    tokenizer.padding_side = "right"

    def log(message: Any) -> None:
        print(message, flush=True)

    train_dataset, eval_dataset, dropped = build_dataset(
        benchmark_cfg, tokenizer, log
    )
    canonical_eval_samples = len(eval_dataset)
    if args.max_eval_samples > 0:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )
    eval_batch_size = int(
        deep_get(benchmark_cfg, "training.eval_batch_size", 8)
    )

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
    result = {
        "schema_version": 1,
        "experiment": "level1_medalpaca_utility_evaluation",
        "status": "completed",
        "run_type": (
            "full"
            if len(eval_dataset) == canonical_eval_samples
            else "smoke"
        ),
        "label": args.label,
        "model": str(deep_get(level1_cfg, "model.id")),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_sha256": (
            sha256_file(adapter_file)
            if adapter_file is not None and adapter_file.is_file()
            else None
        ),
        "dataset": str(deep_get(benchmark_cfg, "dataset.name")),
        "dataset_split": str(deep_get(benchmark_cfg, "dataset.split", "train")),
        "selected_samples": int(deep_get(benchmark_cfg, "dataset.num_samples")),
        "canonical_train_samples": len(train_dataset),
        "canonical_eval_samples": canonical_eval_samples,
        "dropped_zero_response_examples": dropped,
        "max_length": int(deep_get(benchmark_cfg, "dataset.max_length")),
        "eval_batch_size": eval_batch_size,
        "evaluated_samples": len(eval_dataset),
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
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level1-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
