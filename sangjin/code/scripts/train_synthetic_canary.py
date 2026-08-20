#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from opacus import GradSampleModule
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier
from transformers import get_cosine_schedule_with_warmup

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from privacy_eval_common import (  # noqa: E402
    apply_member_canaries,
    canonical_json_sha256,
    deep_get,
    evaluate_response_loss,
    load_selected_raw,
    load_yaml,
    require_cuda_visible_device,
    response_losses,
    set_reproducibility,
    stack_rows,
    tokenize_qa_dataset,
    write_json_exclusive,
)
from train_methods import (  # noqa: E402
    build_model,
    clear_grad_samples,
    clipped_sum_from_grad_samples,
    normalize_grad_sample,
)


def load_and_validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("payload_sha256")
    payload = dict(manifest)
    payload.pop("payload_sha256", None)
    actual_hash = canonical_json_sha256(payload)
    if actual_hash != expected_hash:
        raise ValueError(f"Canary manifest hash mismatch: {actual_hash} != {expected_hash}")
    member_codes = [row["secret_code"] for row in manifest["members"]]
    control_codes = [row["secret_code"] for row in manifest["nonmember_controls"]]
    if len(member_codes) != len(set(member_codes)):
        raise ValueError("Member canary codes are not unique")
    if set(member_codes).intersection(control_codes):
        raise ValueError("Member and nonmember canary codes overlap")
    return manifest


def execute(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path,
    log: Any,
) -> dict[str, Any]:
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    device = require_cuda_visible_device(args.gpu)
    model, tokenizer, parameter_counts = build_model(cfg, device, log)

    selected, dataset_metadata = load_selected_raw(
        str(deep_get(cfg, "dataset.name")),
        int(deep_get(cfg, "dataset.num_samples", 8000)),
        str(deep_get(cfg, "dataset.selection", "shuffled")),
        seed,
    )
    if dataset_metadata != manifest["dataset"]:
        raise ValueError("Canary manifest dataset selection does not match live dataset")
    selected = apply_member_canaries(selected, manifest)
    tokenized, dropped = tokenize_qa_dataset(
        selected, tokenizer, int(deep_get(cfg, "dataset.max_length", 256))
    )
    if dropped != 0:
        raise ValueError(f"Canary experiment unexpectedly dropped {dropped} records")
    train_size = int(deep_get(cfg, "dataset.train_size", 7200))
    eval_size = int(deep_get(cfg, "dataset.eval_size", 800))
    if len(tokenized) != train_size + eval_size:
        raise ValueError("Tokenized dataset size changed")
    train_dataset = tokenized.select(range(train_size))
    eval_dataset = tokenized.select(range(train_size, train_size + eval_size))

    logical_batch = int(deep_get(cfg, "training.logical_batch_size", 128))
    physical_batch = int(deep_get(cfg, "training.physical_batch_size", 16))
    eval_batch = int(deep_get(cfg, "training.eval_batch_size", physical_batch))
    epochs = int(deep_get(cfg, "training.epochs", 6))
    learning_rate = float(deep_get(cfg, "training.learning_rate", 1e-4))
    weight_decay = float(deep_get(cfg, "training.weight_decay", 0.01))
    warmup_steps = int(deep_get(cfg, "training.warmup_steps", 5))
    target_delta = float(deep_get(cfg, "training.target_delta", 1e-5))
    max_grad_norm = float(deep_get(cfg, "training.max_grad_norm", 1.0))
    accountant_name = str(deep_get(cfg, "training.accountant", "prv"))
    logging_steps = int(deep_get(cfg, "training.logging_steps", 10))
    steps_per_epoch = math.ceil(train_size / logical_batch)
    planned_steps = steps_per_epoch * epochs
    steps_to_run = planned_steps if args.max_steps <= 0 else min(args.max_steps, planned_steps)
    sample_rate = logical_batch / train_size

    dp_enabled = args.method == "hooks_dp"
    if args.method not in {"non_dp", "hooks_dp"}:
        raise ValueError(f"Unsupported method: {args.method}")
    if dp_enabled and args.target_epsilon is None:
        raise ValueError("DP canary training requires --target-epsilon")
    if not dp_enabled and args.target_epsilon is not None:
        raise ValueError("non-DP canary training must not receive --target-epsilon")

    noise_multiplier = None
    accountant = None
    if dp_enabled:
        noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=float(args.target_epsilon),
                target_delta=target_delta,
                sample_rate=sample_rate,
                steps=planned_steps,
                accountant=accountant_name,
            )
        )
        accountant = create_accountant(mechanism=accountant_name)
        model = GradSampleModule(
            model, batch_first=True, loss_reduction="mean", strict=False
        )
        log("gradient_mode=opacus_hooks_manual_poisson_dp")
    else:
        log("gradient_mode=standard_non_dp_lora_poisson_batches")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_steps,
    )
    noise_generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    sampling_rng = np.random.default_rng(seed + 20_000)

    resolved = {
        "method": args.method,
        "target_epsilon": args.target_epsilon,
        "target_delta": target_delta if dp_enabled else None,
        "noise_multiplier": noise_multiplier,
        "sample_rate": sample_rate,
        "planned_steps": planned_steps,
        "steps_to_run": steps_to_run,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "dataset": dataset_metadata,
    }
    write_json_exclusive(run_dir / "resolved.json", resolved)
    log(
        f"method={args.method} logical_batch={logical_batch} physical_batch={physical_batch} "
        f"sample_rate={sample_rate:.8f} planned_steps={planned_steps} steps_to_run={steps_to_run}"
    )
    log(
        f"target_epsilon={args.target_epsilon} target_delta={target_delta} "
        f"noise_multiplier={noise_multiplier} max_grad_norm={max_grad_norm}"
    )

    losses: list[float] = []
    step_times: list[float] = []
    lot_sizes: list[int] = []
    total_examples = 0
    epsilon_trace: list[dict[str, Any]] = []
    member_position_to_index = {
        int(row["train_position"]): int(row["canary_index"])
        for row in manifest["members"]
    }
    canary_sample_counts = {
        str(row["canary_index"]): 0 for row in manifest["members"]
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    for step_index in range(steps_to_run):
        step_started = time.perf_counter()
        lot_indices = np.flatnonzero(sampling_rng.random(train_size) < sample_rate)
        if len(lot_indices) == 0:
            raise RuntimeError("Poisson sampler produced an empty lot")
        for sampled_index in lot_indices:
            canary_index = member_position_to_index.get(int(sampled_index))
            if canary_index is not None:
                canary_sample_counts[str(canary_index)] += 1
        optimizer.zero_grad(set_to_none=True)
        lot_loss_sum = 0.0

        if not dp_enabled:
            for start in range(0, len(lot_indices), physical_batch):
                chunk_indices = lot_indices[start : start + physical_batch]
                batch = stack_rows(train_dataset, list(chunk_indices), device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                per_example_loss, _, _ = response_losses(outputs.logits, batch["labels"])
                lot_loss_sum += float(per_example_loss.detach().sum().cpu())
                (per_example_loss.sum() / len(lot_indices)).backward()
        else:
            gradient_buffer = [torch.zeros_like(parameter) for parameter in trainable_parameters]
            for start in range(0, len(lot_indices), physical_batch):
                chunk_indices = lot_indices[start : start + physical_batch]
                batch = stack_rows(train_dataset, list(chunk_indices), device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                per_example_loss, _, _ = response_losses(outputs.logits, batch["labels"])
                lot_loss_sum += float(per_example_loss.detach().sum().cpu())
                per_example_loss.mean().backward()
                grad_samples = []
                for parameter in trainable_parameters:
                    value = getattr(parameter, "grad_sample", None)
                    if value is None:
                        raise RuntimeError("Opacus did not produce grad_sample")
                    grad_samples.append(normalize_grad_sample(value))
                clipped_sums, _ = clipped_sum_from_grad_samples(
                    grad_samples, max_grad_norm
                )
                with torch.no_grad():
                    for buffer, clipped_sum in zip(gradient_buffer, clipped_sums):
                        buffer.add_(clipped_sum)
                clear_grad_samples(trainable_parameters)
                del grad_samples, clipped_sums
            with torch.no_grad():
                for buffer, parameter in zip(gradient_buffer, trainable_parameters):
                    noise = torch.randn(
                        buffer.shape,
                        generator=noise_generator,
                        device=buffer.device,
                        dtype=buffer.dtype,
                    )
                    buffer.add_(noise, alpha=float(noise_multiplier) * max_grad_norm)
                    parameter.grad = buffer.div(logical_batch)
            accountant.step(
                noise_multiplier=float(noise_multiplier), sample_rate=sample_rate
            )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if dp_enabled:
            clear_grad_samples(trainable_parameters)
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - step_started
        average_loss = lot_loss_sum / len(lot_indices)
        losses.append(average_loss)
        step_times.append(elapsed)
        lot_sizes.append(len(lot_indices))
        total_examples += len(lot_indices)
        step_number = step_index + 1
        if step_number == 1 or step_number % logging_steps == 0 or step_number == steps_to_run:
            epsilon_now = (
                float(accountant.get_epsilon(delta=target_delta))
                if accountant is not None
                else None
            )
            epsilon_trace.append({"step": step_number, "epsilon": epsilon_now})
            log(
                f"step={step_number}/{steps_to_run} lot_size={len(lot_indices)} "
                f"loss={average_loss:.6f} lr={scheduler.get_last_lr()[0]:.8g} "
                f"step_time={elapsed:.3f}s epsilon={epsilon_now}"
            )

    torch.cuda.synchronize()
    elapsed_training = time.perf_counter() - started
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    final_epsilon = (
        float(accountant.get_epsilon(delta=target_delta))
        if accountant is not None
        else None
    )
    raw_model = model._module if hasattr(model, "_module") else model
    eval_result = evaluate_response_loss(raw_model, eval_dataset, eval_batch, device)
    adapter_path = run_dir / "final_adapter"
    raw_model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    summary = {
        "schema_version": 1,
        "experiment": "synthetic_canary_training",
        "status": "completed",
        "run_type": "full" if steps_to_run == planned_steps else "smoke",
        "method": args.method,
        "target_epsilon": args.target_epsilon,
        "final_epsilon": final_epsilon,
        "target_delta": target_delta if dp_enabled else None,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm if dp_enabled else None,
        "accountant": accountant_name if dp_enabled else None,
        "manifest_payload_sha256": manifest["payload_sha256"],
        "member_canaries": len(manifest["members"]),
        "nonmember_controls": len(manifest["nonmember_controls"]),
        "canary_sample_counts": canary_sample_counts,
        "canaries_never_sampled": sum(
            count == 0 for count in canary_sample_counts.values()
        ),
        "canary_sample_count_mean": sum(canary_sample_counts.values())
        / len(canary_sample_counts),
        "train_samples": train_size,
        "eval_samples": eval_size,
        "trainable_parameters": parameter_counts[0],
        "logical_batch_size": logical_batch,
        "physical_batch_size": physical_batch,
        "sample_rate": sample_rate,
        "planned_steps": planned_steps,
        "completed_steps": steps_to_run,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "scheduler": "cosine",
        "warmup_steps": warmup_steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "lot_size_mean": sum(lot_sizes) / len(lot_sizes),
        "total_examples_processed": total_examples,
        "elapsed_training_sec": elapsed_training,
        "throughput_samples_per_sec": total_examples / sum(step_times),
        "peak_vram_gb": peak_vram_gb,
        "epsilon_trace": epsilon_trace,
        "eval": eval_result,
        "adapter_path": str(adapter_path),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_visible": args.gpu,
    }
    write_json_exclusive(run_dir / "run_summary.json", summary)
    log(f"completed summary={json.dumps(summary, ensure_ascii=False)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=("non_dp", "hooks_dp"), required=True)
    parser.add_argument("--target-epsilon", type=float)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest = load_and_validate_manifest(args.manifest)
    output_root = Path(str(deep_get(cfg, "paths.canary_training_output_root")))
    run_dir = output_root / args.run_id / args.method
    run_dir.mkdir(parents=True, exist_ok=False)
    training_log = run_dir / "training.log"

    def log(message: Any) -> None:
        print(message, flush=True)
        with training_log.open("a", encoding="utf-8") as stream:
            stream.write(str(message) + "\n")

    try:
        execute(args, cfg, manifest, run_dir, log)
        return 0
    except Exception as error:
        write_json_exclusive(
            run_dir / "run_status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
