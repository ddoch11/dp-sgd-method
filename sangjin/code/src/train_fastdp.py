#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

ROOT = Path(__file__).resolve().parents[1]
FASTDP_ROOT = ROOT / "vendor" / "fast-differential-privacy"
sys.path.insert(0, str(FASTDP_ROOT))

from fastDP import PrivacyEngine as FastDPPrivacyEngine  # noqa: E402
from train_methods import (  # noqa: E402
    build_dataset,
    build_model,
    deep_get,
    evaluate,
    response_losses,
    set_reproducibility,
    stack_rows,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return value


def execute(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path, log: Any) -> dict[str, Any]:
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    log(f"method=fastdp_bk gpu_visible={args.gpu} gpu_name={torch.cuda.get_device_name(0)} seed={seed}")

    model, tokenizer, parameter_counts = build_model(cfg, device, log)
    train_dataset, eval_dataset, dropped_examples = build_dataset(cfg, tokenizer, log)

    logical_batch = int(deep_get(cfg, "training.logical_batch_size", 128))
    physical_batch = int(deep_get(cfg, "training.physical_batch_size", 8))
    eval_batch = int(deep_get(cfg, "training.eval_batch_size", physical_batch))
    epochs = int(deep_get(cfg, "training.epochs", 6))
    learning_rate = float(deep_get(cfg, "training.learning_rate", 1e-4))
    weight_decay = float(deep_get(cfg, "training.weight_decay", 0.01))
    target_delta = float(deep_get(cfg, "training.target_delta", 1e-5))
    max_grad_norm = float(deep_get(cfg, "training.max_grad_norm", 1.0))
    logging_steps = int(deep_get(cfg, "training.logging_steps", 10))
    steps_per_epoch = int(np.ceil(len(train_dataset) / logical_batch))
    planned_steps = steps_per_epoch * epochs
    steps_to_run = planned_steps if args.max_steps <= 0 else min(args.max_steps, planned_steps)
    sample_rate = logical_batch / len(train_dataset)
    sigma = float(
        get_noise_multiplier(
            target_epsilon=args.target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=planned_steps,
            accountant="prv",
        )
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    privacy_engine = FastDPPrivacyEngine(
        model,
        batch_size=logical_batch,
        sample_size=len(train_dataset),
        num_steps=planned_steps,
        noise_multiplier=sigma,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        accounting_mode="rdp",
        clipping_mode="ghost",
        clipping_fn="Abadi",
        clipping_style="all-layer",
        loss_reduction="mean",
    )
    privacy_engine.attach(optimizer)
    prv_accountant = create_accountant(mechanism="prv")
    sampling_rng = np.random.default_rng(seed + 20_000)
    torch.cuda.manual_seed_all(seed + 10_000)

    log(
        f"gradient_mode=fastdp_bookkeeping clipping_mode=ghost clipping_fn=Abadi "
        f"logical_batch={logical_batch} physical_batch={physical_batch} sample_rate={sample_rate:.8f}"
    )
    log(
        f"planned_steps={planned_steps} steps_to_run={steps_to_run} target_epsilon={args.target_epsilon} "
        f"target_delta={target_delta} noise_multiplier={sigma} max_grad_norm={max_grad_norm}"
    )

    losses: list[float] = []
    step_times: list[float] = []
    lot_sizes: list[int] = []
    total_examples = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    for step_index in range(steps_to_run):
        step_started = time.perf_counter()
        lot_indices = np.flatnonzero(sampling_rng.random(len(train_dataset)) < sample_rate)
        if len(lot_indices) == 0:
            raise RuntimeError("Poisson sampler produced an empty lot")
        optimizer.zero_grad(set_to_none=True)
        lot_loss_sum = 0.0
        for start in range(0, len(lot_indices), physical_batch):
            chunk_indices = lot_indices[start : start + physical_batch]
            batch = stack_rows(train_dataset, chunk_indices, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            per_example_loss, _, _ = response_losses(outputs.logits, batch["labels"])
            lot_loss_sum += float(per_example_loss.detach().sum().cpu())
            per_example_loss.mean().backward()
            del batch, outputs, per_example_loss

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        prv_accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        torch.cuda.synchronize()

        step_elapsed = time.perf_counter() - step_started
        step_times.append(step_elapsed)
        lot_sizes.append(len(lot_indices))
        total_examples += len(lot_indices)
        losses.append(lot_loss_sum / len(lot_indices))
        step_number = step_index + 1
        if step_number == 1 or step_number % logging_steps == 0 or step_number == steps_to_run:
            log(
                f"step={step_number}/{steps_to_run} lot_size={len(lot_indices)} "
                f"loss={losses[-1]:.6f} step_time={step_elapsed:.3f}s "
                f"prv_epsilon={prv_accountant.get_epsilon(delta=target_delta):.6f}"
            )

    torch.cuda.synchronize()
    elapsed_training = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    final_epsilon = float(prv_accountant.get_epsilon(delta=target_delta))
    fastdp_privacy = privacy_engine.get_privacy_spent(steps=steps_to_run, lenient=True)
    privacy_engine.detach()

    log("running_final_response_only_eval")
    eval_result = evaluate(model, eval_dataset, eval_batch, device)
    log(f"eval={json.dumps(eval_result, ensure_ascii=False)}")
    adapter_path = run_dir / "final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    summary = {
        "status": "completed",
        "run_type": "full" if steps_to_run == planned_steps else "smoke",
        "method": "fastdp_bk",
        "experiment_date": args.experiment_date,
        "run_dir": str(run_dir),
        "model": str(deep_get(cfg, "model.id")),
        "load_in_4bit": bool(deep_get(cfg, "model.load_in_4bit", False)),
        "attn_implementation": str(deep_get(cfg, "model.attn_implementation", "eager")),
        "dataset": str(deep_get(cfg, "dataset.name")),
        "synthetic_manifest": deep_get(cfg, "dataset.synthetic_manifest"),
        "synthetic_member_samples": int(
            deep_get(cfg, "dataset.synthetic_member_count", 0)
        ),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "dropped_zero_response_examples": dropped_examples,
        "trainable_parameters": parameter_counts[0],
        "total_parameters": parameter_counts[1],
        "seed": seed,
        "logical_batch_size": logical_batch,
        "physical_batch_size": physical_batch,
        "sample_rate": sample_rate,
        "planned_steps": planned_steps,
        "completed_steps": steps_to_run,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "target_epsilon": args.target_epsilon,
        "target_delta": target_delta,
        "accountant": "PRV external; FastDP RDP recorded separately",
        "max_grad_norm": max_grad_norm,
        "noise_multiplier": sigma,
        "final_epsilon": final_epsilon,
        "fastdp_privacy": fastdp_privacy,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "lot_size_mean": sum(lot_sizes) / len(lot_sizes),
        "total_examples_processed": total_examples,
        "elapsed_training_sec": elapsed_training,
        "mean_step_time_sec": sum(step_times) / len(step_times),
        "throughput_samples_per_sec": total_examples / sum(step_times),
        "peak_vram_gb": peak_vram_gb,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_visible": args.gpu,
        "eval": eval_result,
        "adapter_path": str(adapter_path),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"completed final_epsilon={final_epsilon} elapsed_training_sec={elapsed_training:.3f} "
        f"throughput={summary['throughput_samples_per_sec']:.4f} peak_vram_gb={peak_vram_gb:.4f}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-epsilon", type=float, default=2.0)
    parser.add_argument("--experiment-date", default="2026-08-19")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")
    cfg = load_yaml(args.config)
    os.environ.setdefault("HF_HOME", str(deep_get(cfg, "paths.hf_home")))
    os.environ.setdefault("HF_HUB_CACHE", str(deep_get(cfg, "paths.hf_hub_cache")))
    run_dir = (
        Path(str(deep_get(cfg, "paths.output_root")))
        / args.experiment_date
        / "runs"
        / "fastdp_bk"
        / f"eps{args.target_epsilon:g}".replace(".", "p")
        / f"{time.strftime('%Y%m%d_%H%M%S')}_seed{deep_get(cfg, 'runtime.seed', 42)}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    training_log = run_dir / "training.log"

    def log(message: Any) -> None:
        print(message, flush=True)
        with training_log.open("a", encoding="utf-8") as f:
            f.write(str(message) + "\n")

    try:
        execute(args, cfg, run_dir, log)
        return 0
    except Exception as exc:
        (run_dir / "run_status.json").write_text(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
