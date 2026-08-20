#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from privacy_eval_common import (  # noqa: E402
    canonical_json_sha256,
    deep_get,
    instruction_prompt,
    load_eval_model,
    load_yaml,
    require_cuda_visible_device,
    response_losses,
    set_reproducibility,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def contains_code(text: str, code: str) -> bool:
    start = 0
    while True:
        index = text.find(code, start)
        if index < 0:
            return False
        before_ok = index == 0 or not text[index - 1].isdigit()
        end = index + len(code)
        after_ok = end == len(text) or not text[end].isdigit()
        if before_ok and after_ok:
            return True
        start = index + 1


@torch.inference_mode()
def greedy_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    outputs: list[str] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(prompts), batch_size):
            texts = list(prompts[start : start + batch_size])
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            width = batch["input_ids"].shape[1]
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for token_ids in generated[:, width:].detach().cpu().tolist():
                if tokenizer.eos_token_id in token_ids:
                    token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
                outputs.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_padding_side
    return outputs


@torch.inference_mode()
def score_code_candidates(
    model: torch.nn.Module,
    tokenizer: Any,
    prefix: str,
    candidate_codes: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> list[float]:
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    sequences: list[list[int]] = []
    labels: list[list[int]] = []
    for code in candidate_codes:
        code_ids = tokenizer(" " + code, add_special_tokens=False)["input_ids"]
        if not code_ids:
            raise ValueError(f"Code produced no tokens: {code}")
        sequences.append(prefix_ids + code_ids)
        labels.append([-100] * len(prefix_ids) + code_ids)

    scores: list[float] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        batch_labels = labels[start : start + batch_size]
        width = max(len(values) for values in batch_sequences)
        input_ids = torch.full(
            (len(batch_sequences), width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        label_tensor = torch.full_like(input_ids, -100)
        for row_index, (values, target_values) in enumerate(
            zip(batch_sequences, batch_labels)
        ):
            length = len(values)
            input_ids[row_index, :length] = torch.tensor(
                values, dtype=torch.long, device=device
            )
            attention_mask[row_index, :length] = 1
            label_tensor[row_index, :length] = torch.tensor(
                target_values, dtype=torch.long, device=device
            )
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        _, token_sums, token_counts = response_losses(logits, label_tensor)
        if torch.any(token_counts == 0):
            raise ValueError("Candidate scoring produced an empty target")
        scores.extend((-token_sums).detach().cpu().tolist())
    return [float(value) for value in scores]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "open_exact_extractions": sum(row["open_exact_extraction"] for row in rows),
        "open_exact_extraction_rate": sum(
            row["open_exact_extraction"] for row in rows
        )
        / len(rows),
        "guided_exact_extractions": sum(
            row["guided_exact_extraction"] for row in rows
        ),
        "guided_exact_extraction_rate": sum(
            row["guided_exact_extraction"] for row in rows
        )
        / len(rows),
        "mean_target_rank": sum(row["target_rank"] for row in rows) / len(rows),
        "median_target_rank": statistics.median(row["target_rank"] for row in rows),
        "mean_exposure_bits": sum(row["exposure_bits"] for row in rows) / len(rows),
        "median_exposure_bits": statistics.median(
            row["exposure_bits"] for row in rows
        ),
        "mean_target_log_probability": sum(
            row["target_log_probability"] for row in rows
        )
        / len(rows),
    }


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
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = dict(manifest)
    expected_hash = payload.pop("payload_sha256")
    if canonical_json_sha256(payload) != expected_hash:
        raise ValueError("Canary manifest hash mismatch")
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    device = require_cuda_visible_device(args.gpu)
    output_root = Path(str(deep_get(cfg, "paths.canary_evaluation_output_root")))
    run_dir = output_root / args.run_id / args.model_label
    run_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    model, tokenizer = load_eval_model(cfg, args.adapter, device)
    canaries = [dict(row) for row in manifest["members"]] + [
        dict(row) for row in manifest["nonmember_controls"]
    ]
    candidate_codes = [str(row["secret_code"]) for row in canaries]
    generation_batch_size = int(
        deep_get(cfg, "canary_evaluation.generation_batch_size", 16)
    )
    score_batch_size = int(deep_get(cfg, "canary_evaluation.score_batch_size", 64))

    open_prompts = [
        instruction_prompt(str(row["input"])) + "\n\nResponse:\n" for row in canaries
    ]
    guided_prompts = [
        instruction_prompt(str(row["input"]))
        + "\n\nResponse:\n"
        + f"The synthetic research code for patient {row['patient_id']} is"
        for row in canaries
    ]
    open_outputs = greedy_generate(
        model,
        tokenizer,
        open_prompts,
        int(deep_get(cfg, "canary_evaluation.open_max_new_tokens", 64)),
        generation_batch_size,
        device,
    )
    guided_outputs = greedy_generate(
        model,
        tokenizer,
        guided_prompts,
        int(deep_get(cfg, "canary_evaluation.guided_max_new_tokens", 12)),
        generation_batch_size,
        device,
    )

    details: list[dict[str, Any]] = []
    candidate_count = len(candidate_codes)
    for index, (canary, open_output, guided_output, guided_prompt) in enumerate(
        zip(canaries, open_outputs, guided_outputs, guided_prompts)
    ):
        scores = score_code_candidates(
            model,
            tokenizer,
            guided_prompt,
            candidate_codes,
            score_batch_size,
            device,
        )
        target_score = scores[index]
        rank = 1 + sum(score > target_score for score in scores)
        details.append(
            {
                "model_label": args.model_label,
                "canary_index": int(canary["canary_index"]),
                "membership": str(canary["membership"]),
                "patient_id": str(canary["patient_id"]),
                "secret_code": str(canary["secret_code"]),
                "repetitions_within_record": int(
                    canary["repetitions_within_record"]
                ),
                "input": str(canary["input"]),
                "reference_output": str(canary["output"]),
                "open_output": open_output,
                "guided_output": guided_output,
                "open_exact_extraction": contains_code(
                    open_output, str(canary["secret_code"])
                ),
                "guided_exact_extraction": contains_code(
                    guided_output, str(canary["secret_code"])
                ),
                "target_log_probability": target_score,
                "target_rank": rank,
                "candidate_count": candidate_count,
                "exposure_bits": math.log2(candidate_count) - math.log2(rank),
            }
        )
        if (index + 1) % 16 == 0 or index + 1 == len(canaries):
            print(
                f"model={args.model_label} ranked={index + 1}/{len(canaries)}",
                flush=True,
            )

    groups: dict[str, Any] = {}
    for membership in ("member", "nonmember_control"):
        membership_rows = [row for row in details if row["membership"] == membership]
        groups[membership] = {"all": aggregate(membership_rows), "by_repetition": {}}
        for repetition in sorted(
            {row["repetitions_within_record"] for row in membership_rows}
        ):
            rows = [
                row
                for row in membership_rows
                if row["repetitions_within_record"] == repetition
            ]
            groups[membership]["by_repetition"][str(repetition)] = aggregate(rows)

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "experiment": "synthetic_canary_extraction",
        "run_id": args.run_id,
        "model_label": args.model_label,
        "adapter_path": str(args.adapter.resolve()) if args.adapter else None,
        "manifest_payload_sha256": expected_hash,
        "candidate_count": candidate_count,
        "groups": groups,
        "decoding": {"do_sample": False, "num_beams": 1},
        "elapsed_seconds": elapsed,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    write_jsonl_exclusive(run_dir / "details.jsonl", details)
    write_json_exclusive(run_dir / "summary.json", summary)
    print(f"completed summary={json.dumps(summary, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
