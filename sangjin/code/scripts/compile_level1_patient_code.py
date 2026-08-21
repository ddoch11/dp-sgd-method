#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scipy.stats import fisher_exact


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1 - probability) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def rate_text(successes: int, total: int) -> str:
    low, high = wilson(successes, total)
    return (
        f"{successes}/{total} ({100 * successes / total:.2f}%, "
        f"95% CI {100 * low:.2f}-{100 * high:.2f}%)"
    )


def extraction_statistics(summary: dict[str, Any]) -> dict[str, Any]:
    member = summary["groups"]["member"]
    control = summary["groups"]["control"]
    member_exact = int(member["exact_extractions"])
    control_exact = int(control["exact_extractions"])
    member_total = int(member["samples"])
    control_total = int(control["samples"])
    return {
        "member_exact": member_exact,
        "control_exact": control_exact,
        "member_total": member_total,
        "control_total": control_total,
        "member_control_exact_excess": member_exact / member_total
        - control_exact / control_total,
        "fisher_two_sided_p": fisher_exact(
            [
                [member_exact, member_total - member_exact],
                [control_exact, control_total - control_exact],
            ],
            alternative="two-sided",
        ).pvalue,
        "member_mean_target_log_probability": member["mean_target_log_probability"],
        "control_mean_target_log_probability": control["mean_target_log_probability"],
        "target_score_membership_auc": summary["target_score_membership_auc"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    runs = args.results_root / "runs"
    base_summary_path = runs / "evaluation/base_20260821/base/summary.json"
    base_details_path = runs / "evaluation/base_20260821/base/details.jsonl"
    non_dp_root = runs / "training/pilot_non_dp_20260821/non_dp"
    dp_roots = {
        "dp_eps0p5": runs / "training/full_dp_20260821/opacus_hooks_dp_eps0p5",
        "dp_eps2": runs / "training/full_dp_20260821/opacus_hooks_dp_eps2",
        "dp_eps8": runs / "training/full_dp_20260821/opacus_hooks_dp_eps8",
    }

    base_summary = load_json(base_summary_path)
    base_details = load_jsonl(base_details_path)
    pilot_epochs = [1, 5, 10, 20]
    pilot_summaries = {
        str(epoch): load_json(
            non_dp_root
            / f"checkpoints/epoch_{epoch:03d}/evaluation/summary.json"
        )
        for epoch in pilot_epochs
    }
    pilot_details = load_jsonl(
        non_dp_root / "checkpoints/epoch_020/evaluation/details.jsonl"
    )
    training = {"non_dp": load_json(non_dp_root / "run_summary.json")}
    final_summaries = {
        "base": base_summary,
        "non_dp": pilot_summaries["20"],
    }
    final_details = {"base": base_details, "non_dp": pilot_details}
    for label, root in dp_roots.items():
        training[label] = load_json(root / "run_summary.json")
        final_summaries[label] = load_json(
            root / "checkpoints/epoch_020/evaluation/summary.json"
        )
        final_details[label] = load_jsonl(
            root / "checkpoints/epoch_020/evaluation/details.jsonl"
        )

    statistics = {
        label: extraction_statistics(summary)
        for label, summary in final_summaries.items()
    }
    pilot_statistics = {
        epoch: extraction_statistics(summary)
        for epoch, summary in pilot_summaries.items()
    }

    display = {
        "base": "Base",
        "non_dp": "non-DP LoRA",
        "dp_eps0p5": "Opacus DP epsilon=0.5",
        "dp_eps2": "Opacus DP epsilon=2",
        "dp_eps8": "Opacus DP epsilon=8",
    }
    lines = [
        "# 2026-08-21 Level 1 합성 환자 코드 실험",
        "",
        "> 실제 개인정보를 사용하지 않았다. 1,000개의 합성 환자 alias와 무작위 네 자리 private code를 사용한 BF16 Opacus DP-SGD memorization 실험이다.",
        "",
        "## 설정",
        "",
        "- Model: VaultGemma-1B BF16 + LoRA r=8, alpha=16, dropout=0",
        "- Member 500개: fine-tuning에 포함",
        "- Control 500개: fine-tuning에 미포함",
        "- Prompt: `What is the private code for synthetic patient SYN-Pxxxx?`",
        "- Target: 무작위 고유 네 자리 코드만 출력",
        "- Max length 64, logical/physical batch 32/16",
        "- non-DP pilot 20 epochs, 320 optimizer steps",
        "- DP: Opacus PrivacyEngine + Hooks + DPOptimizer + BMM + PRV, Poisson sampling",
        "- delta=1e-5, C=1, seed=42",
        "",
        "## non-DP positive-control pilot",
        "",
        "| Epoch | Member exact | Control exact | Exact excess | Score AUC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for epoch in pilot_epochs:
        stats = pilot_statistics[str(epoch)]
        lines.append(
            f"| {epoch} | {stats['member_exact']}/500 | {stats['control_exact']}/500 | "
            f"{100 * stats['member_control_exact_excess']:+.2f}%p | "
            f"{stats['target_score_membership_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 최종 모델 비교",
            "",
            "| 모델 | Target/Actual epsilon | Noise sigma | Member exact | Control exact | Exact excess | Fisher p | Score AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("base", "non_dp", "dp_eps0p5", "dp_eps2", "dp_eps8"):
        stats = statistics[label]
        if label == "base":
            epsilon_text = "-"
            sigma_text = "-"
        elif label == "non_dp":
            epsilon_text = "-"
            sigma_text = "-"
        else:
            run = training[label]
            epsilon_text = f"{run['target_epsilon']:g}/{run['final_epsilon']:.4f}"
            sigma_text = f"{run['noise_multiplier']:.6f}"
        lines.append(
            f"| {display[label]} | {epsilon_text} | {sigma_text} | "
            f"{rate_text(stats['member_exact'], 500)} | "
            f"{rate_text(stats['control_exact'], 500)} | "
            f"{100 * stats['member_control_exact_excess']:+.2f}%p | "
            f"{stats['fisher_two_sided_p']:.4f} | "
            f"{stats['target_score_membership_auc']:.4f} |"
        )

    indexes = {
        label: {row["patient_id"]: row for row in rows}
        for label, rows in final_details.items()
    }
    example_ids = sorted(
        row["patient_id"]
        for row in final_details["non_dp"]
        if row["membership"] == "member" and row["exact_extraction"]
    )[:5]
    lines.extend(
        [
            "",
            "## 정성 예시",
            "",
            "| Patient | Target | Base | non-DP | DP epsilon=2 |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for patient_id in example_ids:
        lines.append(
            f"| {patient_id} | {indexes['non_dp'][patient_id]['private_code']} | "
            f"{indexes['base'][patient_id]['generated_text']} | "
            f"{indexes['non_dp'][patient_id]['generated_text']} | "
            f"{indexes['dp_eps2'][patient_id]['generated_text']} |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- Base는 Member/Control code를 전혀 맞히지 못했고 AUC도 0.5 부근이었다.",
            "- non-DP는 epoch 10부터 target score AUC가 0.8을 넘었고 epoch 20에는 Member code 10개를 exact 추출했으며 Control exact는 0개였다.",
            "- Opacus DP epsilon=0.5/2/8은 모두 Member·Control exact 0개, score AUC 0.5 부근이었다.",
            "- 따라서 이 Level 1 조건에서는 일반 LoRA가 합성 환자-코드 mapping을 암기했지만 DP-SGD에서는 동일 공격 신호가 탐지되지 않았다.",
            "- 실제 환자정보 유출을 측정한 것이 아니라 통제된 합성 memorization stress다.",
            "- 단일 seed와 secure RNG 비활성화 실험이며, 최종 주장 전 seed 반복이 필요하다.",
            "",
        ]
    )

    combined = {
        "schema_version": 1,
        "manifest_payload_sha256": base_summary["manifest_payload_sha256"],
        "pilot_summaries": pilot_summaries,
        "pilot_statistics": pilot_statistics,
        "training": training,
        "final_summaries": final_summaries,
        "statistics": statistics,
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
