#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from level1_patient_code_common import (  # noqa: E402
    deep_get,
    evaluate_model,
    load_eval_model,
    load_manifest,
    load_yaml,
    require_cuda_visible_device,
    set_reproducibility,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest = load_manifest(args.manifest)
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    device = require_cuda_visible_device(args.gpu)
    output_root = Path(str(deep_get(cfg, "paths.output_root")))
    run_dir = output_root / "evaluation" / args.run_id / args.model_label
    run_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    model, tokenizer = load_eval_model(cfg, args.adapter, device)
    summary, details = evaluate_model(
        model, tokenizer, list(manifest["records"]), cfg, device
    )
    summary.update(
        {
            "schema_version": 1,
            "experiment": "level1_patient_code_evaluation",
            "run_id": args.run_id,
            "model_label": args.model_label,
            "adapter_path": str(args.adapter.resolve()) if args.adapter else None,
            "manifest_payload_sha256": manifest["payload_sha256"],
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": torch.cuda.get_device_name(0),
        }
    )
    write_jsonl_exclusive(run_dir / "details.jsonl", details)
    write_json_exclusive(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
