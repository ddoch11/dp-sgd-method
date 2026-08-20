#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from privacy_eval_common import (  # noqa: E402
    canonical_json_sha256,
    deep_get,
    instruction_prompt,
    load_eval_model,
    load_selected_raw,
    load_yaml,
    matching_prefix_length,
    require_cuda_visible_device,
    response_text,
    set_reproducibility,
    token_edit_distance,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def build_cases(
    raw: Any,
    tokenizer: Any,
    train_size: int,
    max_length: int,
    profile: dict[str, Any],
    selection_seed: int,
    excluded_selected_positions: set[int],
) -> list[dict[str, Any]]:
    split = str(profile["split"])
    if split == "member":
        start, stop = 0, train_size
    elif split == "nonmember":
        start, stop = train_size, len(raw)
    else:
        raise ValueError(f"Unsupported profile split: {split}")
    prefix_tokens = int(profile["prefix_tokens"])
    suffix_tokens = int(profile["suffix_tokens"])
    eligible: list[dict[str, Any]] = []
    for selected_position in range(start, stop):
        if selected_position in excluded_selected_positions:
            continue
        row = raw[selected_position]
        prompt = instruction_prompt(str(row["input"]))
        full_text = prompt + response_text(str(row["output"]))
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        # Match the training pipeline exactly: it uses the separately tokenized
        # prompt length to mask the combined sequence, even when a boundary token
        # changes after concatenating the response text.
        response_ids = full_ids[len(prompt_ids) :]
        if len(response_ids) < prefix_tokens + suffix_tokens:
            continue
        input_ids = full_ids[: len(prompt_ids) + prefix_tokens]
        target_ids = response_ids[prefix_tokens : prefix_tokens + suffix_tokens]
        eligible.append(
            {
                "selected_position": selected_position,
                "source_index": int(row["_source_index"]),
                "question": str(row["input"]),
                "reference_answer": str(row["output"]),
                "input_ids": input_ids,
                "target_ids": target_ids,
                "response_token_count": len(response_ids),
            }
        )
    sample_count = int(profile["sample_count"])
    if len(eligible) < sample_count:
        raise ValueError(
            f"Profile {profile['name']} has {len(eligible)} eligible rows, needs {sample_count}"
        )
    rng = random.Random(selection_seed)
    rng.shuffle(eligible)
    return eligible[:sample_count]


@torch.inference_mode()
def generate_cases(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    suffix_tokens: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start : start + batch_size]
        width = max(len(case["input_ids"]) for case in batch_cases)
        input_ids = torch.full(
            (len(batch_cases), width),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, case in enumerate(batch_cases):
            values = torch.tensor(case["input_ids"], dtype=torch.long, device=device)
            input_ids[row_index, -len(values) :] = values
            attention_mask[row_index, -len(values) :] = 1
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=suffix_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_suffixes = generated[:, width:].detach().cpu().tolist()
        for case, generated_ids in zip(batch_cases, generated_suffixes):
            if tokenizer.eos_token_id in generated_ids:
                generated_ids = generated_ids[: generated_ids.index(tokenizer.eos_token_id)]
            target_ids = list(case["target_ids"])
            distance = token_edit_distance(generated_ids, target_ids)
            denominator = max(len(generated_ids), len(target_ids), 1)
            positional_matches = sum(
                left == right for left, right in zip(generated_ids, target_ids)
            )
            rows.append(
                {
                    **{key: value for key, value in case.items() if key not in {"input_ids", "target_ids"}},
                    "input_text": tokenizer.decode(case["input_ids"], skip_special_tokens=True),
                    "target_suffix": tokenizer.decode(target_ids, skip_special_tokens=True),
                    "generated_suffix": tokenizer.decode(generated_ids, skip_special_tokens=True),
                    "target_token_ids": target_ids,
                    "generated_token_ids": generated_ids,
                    "exact_match": generated_ids == target_ids,
                    "edit_distance": distance,
                    "normalized_edit_similarity": 1.0 - (distance / denominator),
                    "approximate_match_10pct": distance <= max(1, int(0.1 * len(target_ids))),
                    "matching_prefix_tokens": matching_prefix_length(generated_ids, target_ids),
                    "positional_token_accuracy": positional_matches / len(target_ids),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "exact_matches": sum(bool(row["exact_match"]) for row in rows),
        "exact_match_rate": sum(bool(row["exact_match"]) for row in rows) / len(rows),
        "approximate_matches_10pct": sum(
            bool(row["approximate_match_10pct"]) for row in rows
        ),
        "approximate_match_rate_10pct": sum(
            bool(row["approximate_match_10pct"]) for row in rows
        )
        / len(rows),
        "mean_normalized_edit_similarity": sum(
            float(row["normalized_edit_similarity"]) for row in rows
        )
        / len(rows),
        "mean_matching_prefix_tokens": sum(
            int(row["matching_prefix_tokens"]) for row in rows
        )
        / len(rows),
        "mean_positional_token_accuracy": sum(
            float(row["positional_token_accuracy"]) for row in rows
        )
        / len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection", choices=("head", "shuffled"))
    parser.add_argument("--exclude-canary-manifest", type=Path)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(deep_get(cfg, "runtime.seed", 42))
    set_reproducibility(seed)
    device = require_cuda_visible_device(args.gpu)
    output_root = Path(str(deep_get(cfg, "paths.prefix_suffix_output_root")))
    run_dir = output_root / args.run_id / args.model_label
    run_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    model, tokenizer = load_eval_model(cfg, args.adapter, device)
    selection = args.selection or str(deep_get(cfg, "prefix_suffix.selection", "head"))
    raw, dataset_metadata = load_selected_raw(
        str(deep_get(cfg, "dataset.name")),
        int(deep_get(cfg, "dataset.num_samples", 8000)),
        selection,
        seed,
    )
    excluded_selected_positions: set[int] = set()
    excluded_manifest_hash = None
    if args.exclude_canary_manifest is not None:
        manifest = json.loads(
            args.exclude_canary_manifest.read_text(encoding="utf-8")
        )
        excluded_manifest_hash = manifest.get("payload_sha256")
        if manifest.get("dataset") != dataset_metadata:
            raise ValueError("Excluded Canary manifest uses a different dataset selection")
        excluded_selected_positions = {
            int(row["train_position"]) for row in manifest["members"]
        }
    train_size = int(deep_get(cfg, "dataset.train_size", 7200))
    max_length = int(deep_get(cfg, "dataset.max_length", 256))
    batch_size = int(deep_get(cfg, "prefix_suffix.generation_batch_size", 8))
    profile_summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for profile_index, raw_profile in enumerate(deep_get(cfg, "prefix_suffix.profiles")):
        profile = dict(raw_profile)
        cases = build_cases(
            raw,
            tokenizer,
            train_size,
            max_length,
            profile,
            selection_seed=seed + 1000 * (profile_index + 1),
            excluded_selected_positions=excluded_selected_positions,
        )
        rows = generate_cases(
            model,
            tokenizer,
            cases,
            int(profile["suffix_tokens"]),
            batch_size,
            device,
        )
        for row in rows:
            row["profile"] = str(profile["name"])
            row["split"] = str(profile["split"])
            row["model_label"] = args.model_label
        all_rows.extend(rows)
        profile_summaries[str(profile["name"])] = {
            "definition": profile,
            "case_source_indices_sha256": canonical_json_sha256(
                [row["source_index"] for row in rows]
            ),
            **summarize(rows),
        }
        print(
            f"profile={profile['name']} model={args.model_label} "
            f"summary={json.dumps(profile_summaries[str(profile['name'])], ensure_ascii=False)}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "experiment": "prefix_suffix_extraction",
        "run_id": args.run_id,
        "model_label": args.model_label,
        "model_id": str(deep_get(cfg, "model.id")),
        "adapter_path": str(args.adapter.resolve()) if args.adapter else None,
        "dataset": dataset_metadata,
        "excluded_selected_positions": len(excluded_selected_positions),
        "excluded_canary_manifest_sha256": excluded_manifest_hash,
        "seed": seed,
        "decoding": {"do_sample": False, "num_beams": 1},
        "profiles": profile_summaries,
        "elapsed_seconds": elapsed,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    write_jsonl_exclusive(run_dir / "details.jsonl", all_rows)
    write_json_exclusive(run_dir / "summary.json", summary)
    print(f"completed model={args.model_label} elapsed={elapsed:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
