#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compile_level1_patient_code import (
    extraction_statistics,
    load_json,
    load_jsonl,
    rate_text,
)


NON_DP_RUNS = {
    "lr1e4_e20": ("pilot_non_dp_20260821", 20),
    "lr3e4_e20": ("pilot_grid_lr3e4_e20_20260821", 20),
    "lr1e4_e40": ("pilot_grid_lr1e4_e40_20260821", 40),
    "lr3e4_e40": ("pilot_grid_lr3e4_e40_20260821", 40),
    "lr1e4_e80": ("pilot_grid_lr1e4_e80_20260821", 80),
}
OLD_DP_RUN_ID = "full_dp_20260821"
TUNED_DP_RUN_ID = "tuned_dp_lr1e4_e40_20260821"
DP_LABELS = ("dp_eps0p5", "dp_eps2", "dp_eps8")


def checkpoint_root(run_root: Path, epoch: int) -> Path:
    return run_root / "checkpoints" / f"epoch_{epoch:03d}" / "evaluation"


def dp_method_dir(label: str) -> str:
    return f"opacus_hooks_dp_eps{label.removeprefix('dp_eps')}"


def md_code(value: str) -> str:
    escaped = value.replace("|", "\\|").replace("`", "\\`")
    return f"`{escaped}`"


def load_run(run_root: Path, epoch: int) -> dict[str, Any]:
    evaluation_root = checkpoint_root(run_root, epoch)
    summary = load_json(evaluation_root / "summary.json")
    return {
        "training": load_json(run_root / "run_summary.json"),
        "evaluation": summary,
        "statistics": extraction_statistics(summary),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    runs = args.results_root / "runs"
    base_root = runs / "evaluation" / "base_20260821" / "base"
    base_summary = load_json(base_root / "summary.json")
    base_details = load_jsonl(base_root / "details.jsonl")

    non_dp_grid: dict[str, dict[str, Any]] = {}
    for label, (run_id, epoch) in NON_DP_RUNS.items():
        run_root = runs / "training" / run_id / "non_dp"
        non_dp_grid[label] = load_run(run_root, epoch)

    selected_non_dp_root = (
        runs / "training" / NON_DP_RUNS["lr1e4_e40"][0] / "non_dp"
    )
    selected_non_dp = non_dp_grid["lr1e4_e40"]
    selected_non_dp_details = load_jsonl(
        checkpoint_root(selected_non_dp_root, 40) / "details.jsonl"
    )

    dp_runs: dict[str, dict[str, Any]] = {}
    old_dp_training: dict[str, dict[str, Any]] = {}
    dp_details: dict[str, list[dict[str, Any]]] = {}
    for label in DP_LABELS:
        method_dir = dp_method_dir(label)
        tuned_root = runs / "training" / TUNED_DP_RUN_ID / method_dir
        old_root = runs / "training" / OLD_DP_RUN_ID / method_dir
        dp_runs[label] = load_run(tuned_root, 40)
        old_dp_training[label] = load_json(old_root / "run_summary.json")
        dp_details[label] = load_jsonl(
            checkpoint_root(tuned_root, 40) / "details.jsonl"
        )

    final_summaries = {
        "base": base_summary,
        "non_dp": selected_non_dp["evaluation"],
        **{label: run["evaluation"] for label, run in dp_runs.items()},
    }
    statistics = {
        label: extraction_statistics(summary)
        for label, summary in final_summaries.items()
    }
    training = {
        "non_dp": selected_non_dp["training"],
        **{label: run["training"] for label, run in dp_runs.items()},
    }

    details = {
        "base": {row["patient_id"]: row for row in base_details},
        "non_dp": {row["patient_id"]: row for row in selected_non_dp_details},
        **{
            label: {row["patient_id"]: row for row in rows}
            for label, rows in dp_details.items()
        },
    }
    example_ids = sorted(
        row["patient_id"]
        for row in selected_non_dp_details
        if row["membership"] == "member" and row["exact_extraction"]
    )[:5]
    qualitative_examples = [
        {
            "patient_id": patient_id,
            "target": details["non_dp"][patient_id]["private_code"],
            **{
                label: details[label][patient_id]["generated_text"]
                for label in ("base", "non_dp", *DP_LABELS)
            },
        }
        for patient_id in example_ids
    ]

    lines = [
        "# 2026-08-21 Level 1 합성 환자 코드 40 epoch 재실험",
        "",
        "> 20 epoch non-DP의 직접 복원율이 2%에 그쳐 positive control이 약하다는 판단에 따라 학습 강도를 재탐색하고, 선택한 조건으로 DP 모델을 다시 학습했다. 실제 개인정보는 사용하지 않았다.",
        "",
        "## 재실험 사유",
        "",
        "- 기존 20 epoch non-DP는 Score AUC 0.9983이었지만 Member exact는 10/500에 불과했다.",
        "- AUC는 정답 확률의 Member/Control 분리를 뜻할 뿐, 모델이 코드를 안정적으로 생성한다는 뜻은 아니다.",
        "- 따라서 direct extraction이 충분히 발생하는 non-DP 조건을 먼저 찾은 뒤 같은 조건으로 DP를 비교했다.",
        "",
        "## 고정 조건",
        "",
        "- VaultGemma-1B BF16 + LoRA r=8, alpha=16, dropout=0",
        "- 합성 Member 500개 학습, 합성 Control 500개 미학습",
        "- 무작위 고유 네 자리 code, 한 환자당 한 record, 중복 없음",
        "- Max length 64, logical/physical batch 32/16",
        "- AdamW, weight decay 0, constant scheduler, seed 42",
        "- DP: Opacus Hooks, Poisson sampling, PRV accountant, delta=1e-5, C=1",
        "",
        "## non-DP 학습 강도 탐색",
        "",
        "| Learning rate | Epoch | Steps | Final train loss | Member exact | Control exact | Score AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grid_order = (
        "lr1e4_e20",
        "lr3e4_e20",
        "lr1e4_e40",
        "lr3e4_e40",
        "lr1e4_e80",
    )
    for label in grid_order:
        run = non_dp_grid[label]
        training_row = run["training"]
        stats = run["statistics"]
        lines.append(
            f"| {training_row['learning_rate']:.0e} | {training_row['epochs']} | "
            f"{training_row['completed_steps']} | {training_row['loss_last']:.4f} | "
            f"{stats['member_exact']}/500 | {stats['control_exact']}/500 | "
            f"{stats['target_score_membership_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`lr=1e-4, 40 epoch`은 Member exact 97.6%, Control exact 0%로 task 학습이 명확하고, 80 epoch보다 step이 절반이므로 최종 비교 조건으로 선택했다.",
            "",
            "## 고정 epsilon에서 epoch 증가가 noise에 미치는 영향",
            "",
            "| Target epsilon | sigma at 20 epoch | sigma at 40 epoch | 증가율 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for label in DP_LABELS:
        old_run = old_dp_training[label]
        new_run = dp_runs[label]["training"]
        increase = new_run["noise_multiplier"] / old_run["noise_multiplier"] - 1
        lines.append(
            f"| {new_run['target_epsilon']:g} | {old_run['noise_multiplier']:.6f} | "
            f"{new_run['noise_multiplier']:.6f} | {100 * increase:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "Privacy budget을 고정한 채 optimizer step을 320에서 640으로 늘렸기 때문에 accountant가 요구하는 noise multiplier도 커졌다.",
            "",
            "## 40 epoch 최종 비교",
            "",
            "| 모델 | Target / Actual epsilon | Noise sigma | Final train loss | Member exact | Control exact | Score AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    display = {
        "base": "Base",
        "non_dp": "non-DP LoRA",
        "dp_eps0p5": "Opacus DP epsilon=0.5",
        "dp_eps2": "Opacus DP epsilon=2",
        "dp_eps8": "Opacus DP epsilon=8",
    }
    for label in ("base", "non_dp", *DP_LABELS):
        stats = statistics[label]
        if label == "base":
            epsilon_text = sigma_text = loss_text = "-"
        elif label == "non_dp":
            epsilon_text = sigma_text = "-"
            loss_text = f"{training[label]['loss_last']:.4f}"
        else:
            run = training[label]
            epsilon_text = f"{run['target_epsilon']:g}/{run['final_epsilon']:.4f}"
            sigma_text = f"{run['noise_multiplier']:.6f}"
            loss_text = f"{run['loss_last']:.4f}"
        lines.append(
            f"| {display[label]} | {epsilon_text} | {sigma_text} | {loss_text} | "
            f"{rate_text(stats['member_exact'], 500)} | "
            f"{rate_text(stats['control_exact'], 500)} | "
            f"{stats['target_score_membership_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 정성 예시",
            "",
            "| Patient | Target | Base | non-DP | DP epsilon=0.5 | DP epsilon=2 | DP epsilon=8 |",
            "|---|---:|---|---|---|---|---|",
        ]
    )
    for row in qualitative_examples:
        lines.append(
            f"| {row['patient_id']} | {row['target']} | {md_code(row['base'])} | "
            f"{md_code(row['non_dp'])} | {md_code(row['dp_eps0p5'])} | "
            f"{md_code(row['dp_eps2'])} | {md_code(row['dp_eps8'])} |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 20 epoch의 2% exact는 task를 충분히 학습한 positive control로 보기 어려웠다.",
            "- 40 epoch non-DP는 Member 488/500을 복원하고 Control은 0/500이어서 memorization positive control이 명확히 성립했다.",
            "- 같은 40 epoch의 DP epsilon=0.5와 2는 Member exact 0건, epsilon=8은 1건이었고 Control은 모두 0건이었다.",
            "- Score AUC도 DP 세 모델에서 0.48-0.53으로 무작위 수준에 가까웠다.",
            "- 이는 이 공격과 단일 seed 조건에서 DP 모델의 memorization 신호가 크게 억제됐다는 경험적 결과다. 모든 개인정보 공격을 막았다는 뜻은 아니다.",
            "- 이 task는 합성 key-value memorization stress이므로 일반 의료 QA utility를 평가하지 않는다.",
            "- secure RNG를 끈 실험이며, 최종 통계 주장을 위해서는 seed 반복과 별도 공격 평가가 필요하다.",
            "",
        ]
    )

    combined = {
        "schema_version": 1,
        "report_variant": "level1_patient_code_tuned_40_epoch",
        "manifest_payload_sha256": base_summary["manifest_payload_sha256"],
        "selected_recipe": {
            "learning_rate": 0.0001,
            "epochs": 40,
            "logical_steps": 640,
        },
        "non_dp_grid": non_dp_grid,
        "old_dp_training": old_dp_training,
        "training": training,
        "final_summaries": final_summaries,
        "statistics": statistics,
        "qualitative_examples": qualitative_examples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(combined, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    with args.output_md.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(f"created {args.output_json} and {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
