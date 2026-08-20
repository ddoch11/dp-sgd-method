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
    lower, upper = wilson(successes, total)
    return (
        f"{successes}/{total} ({100 * successes / total:.2f}%, "
        f"95% CI {100 * lower:.2f}-{100 * upper:.2f}%)"
    )


def pairwise_auc(member_scores: list[float], control_scores: list[float]) -> float:
    total = 0.0
    for member_score in member_scores:
        for control_score in control_scores:
            if member_score > control_score:
                total += 1.0
            elif member_score == control_score:
                total += 0.5
    return total / (len(member_scores) * len(control_scores))


def prefix_statistics(summary: dict[str, Any]) -> dict[str, Any]:
    member = summary["profiles"]["qa_member_10_20"]
    control = summary["profiles"]["qa_nonmember_10_20"]
    member_total = int(member["samples"])
    control_total = int(control["samples"])
    member_exact = int(member["exact_matches"])
    control_exact = int(control["exact_matches"])
    member_approx = int(member["approximate_matches_10pct"])
    control_approx = int(control["approximate_matches_10pct"])
    exact_p = fisher_exact(
        [
            [member_exact, member_total - member_exact],
            [control_exact, control_total - control_exact],
        ],
        alternative="two-sided",
    ).pvalue
    approx_p = fisher_exact(
        [
            [member_approx, member_total - member_approx],
            [control_approx, control_total - control_approx],
        ],
        alternative="two-sided",
    ).pvalue
    return {
        "member_exact": member_exact,
        "control_exact": control_exact,
        "member_total": member_total,
        "control_total": control_total,
        "exact_member_excess": member_exact / member_total
        - control_exact / control_total,
        "exact_fisher_two_sided_p": exact_p,
        "member_approximate": member_approx,
        "control_approximate": control_approx,
        "approximate_member_excess": member_approx / member_total
        - control_approx / control_total,
        "approximate_fisher_two_sided_p": approx_p,
        "member_mean_edit_similarity": member["mean_normalized_edit_similarity"],
        "control_mean_edit_similarity": control["mean_normalized_edit_similarity"],
    }


def canary_statistics(
    summary: dict[str, Any], details: list[dict[str, Any]]
) -> dict[str, Any]:
    member = summary["groups"]["member"]["all"]
    control = summary["groups"]["nonmember_control"]["all"]
    member_rows = [row for row in details if row["membership"] == "member"]
    control_rows = [
        row for row in details if row["membership"] == "nonmember_control"
    ]
    score_auc = pairwise_auc(
        [float(row["target_log_probability"]) for row in member_rows],
        [float(row["target_log_probability"]) for row in control_rows],
    )
    return {
        "member_open_exact": int(member["open_exact_extractions"]),
        "control_open_exact": int(control["open_exact_extractions"]),
        "member_guided_exact": int(member["guided_exact_extractions"]),
        "control_guided_exact": int(control["guided_exact_extractions"]),
        "member_exposure_bits": float(member["mean_exposure_bits"]),
        "control_exposure_bits": float(control["mean_exposure_bits"]),
        "exposure_gap_bits": float(member["mean_exposure_bits"])
        - float(control["mean_exposure_bits"]),
        "member_mean_rank": float(member["mean_target_rank"]),
        "control_mean_rank": float(control["mean_target_rank"]),
        "member_target_log_probability": float(
            member["mean_target_log_probability"]
        ),
        "control_target_log_probability": float(
            control["mean_target_log_probability"]
        ),
        "target_log_probability_gap": float(member["mean_target_log_probability"])
        - float(control["mean_target_log_probability"]),
        "target_score_membership_auc": score_auc,
    }


def markdown_cell(value: Any, limit: int = 170) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_prefix_group(
    run_dir: Path, labels: list[str]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summaries = {
        label: load_json(run_dir / label / "summary.json") for label in labels
    }
    details = {
        label: load_jsonl(run_dir / label / "details.jsonl") for label in labels
    }
    return summaries, details


def load_canary_group(
    run_dir: Path, labels: list[str]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    return load_prefix_group(run_dir, labels)


def append_prefix_table(
    lines: list[str],
    title: str,
    summaries: dict[str, Any],
    labels: list[str],
) -> dict[str, Any]:
    lines.extend(
        [
            f"### {title}",
            "",
            "| 모델 | Member exact | Control exact | Exact excess | Fisher p | Member approx | Control approx | Approx p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    statistics: dict[str, Any] = {}
    for label in labels:
        stats = prefix_statistics(summaries[label])
        statistics[label] = stats
        lines.append(
            f"| {label} | {rate_text(stats['member_exact'], stats['member_total'])} | "
            f"{rate_text(stats['control_exact'], stats['control_total'])} | "
            f"{100 * stats['exact_member_excess']:+.2f}%p | "
            f"{stats['exact_fisher_two_sided_p']:.4f} | "
            f"{stats['member_approximate']}/{stats['member_total']} | "
            f"{stats['control_approximate']}/{stats['control_total']} | "
            f"{stats['approximate_fisher_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| 모델 | Long 50->50 exact | Long approximate <=10% | Mean edit similarity |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in labels:
        result = summaries[label]["profiles"]["vaultgemma_member_50_50"]
        lines.append(
            f"| {label} | {result['exact_matches']}/{result['samples']} | "
            f"{result['approximate_matches_10pct']}/{result['samples']} | "
            f"{result['mean_normalized_edit_similarity']:.4f} |"
        )
    lines.append("")
    return statistics


def append_canary_table(
    lines: list[str],
    title: str,
    summaries: dict[str, Any],
    details: dict[str, list[dict[str, Any]]],
    labels: list[str],
) -> dict[str, Any]:
    lines.extend(
        [
            f"### {title}",
            "",
            "| 모델 | Member guided exact | Control guided exact | Member exposure | Control exposure | Gap | Score AUC | Member rank |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    statistics: dict[str, Any] = {}
    for label in labels:
        stats = canary_statistics(summaries[label], details[label])
        statistics[label] = stats
        lines.append(
            f"| {label} | {stats['member_guided_exact']}/64 | "
            f"{stats['control_guided_exact']}/64 | "
            f"{stats['member_exposure_bits']:.3f} | "
            f"{stats['control_exposure_bits']:.3f} | "
            f"{stats['exposure_gap_bits']:+.3f} | "
            f"{stats['target_score_membership_auc']:.3f} | "
            f"{stats['member_mean_rank']:.2f} |"
        )
    lines.append("")
    return statistics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    runs = args.results_root / "runs"
    legacy_labels = ["base", "non_dp", "dp_eps2_hooks"]
    canonical_labels = [
        "base",
        "non_dp_canary",
        "dp_eps0p5_canary",
        "dp_eps2_canary",
        "dp_eps8_canary",
    ]
    standard_canary_labels = canonical_labels
    stress_canary_labels = ["stress_base", "stress_non_dp", "stress_dp_eps2"]
    legacy_summaries, legacy_details = load_prefix_group(
        runs / "prefix_suffix" / "existing_4bit_20260820_n196", legacy_labels
    )
    canonical_summaries, canonical_details = load_prefix_group(
        runs / "prefix_suffix" / "canonical_canary_models_20260820",
        canonical_labels,
    )
    standard_canary_summaries, standard_canary_details = load_canary_group(
        runs / "canary_evaluation" / "canary_full_20260820",
        standard_canary_labels,
    )
    stress_canary_summaries, stress_canary_details = load_canary_group(
        runs / "canary_evaluation" / "canary_stress_20260820",
        stress_canary_labels,
    )

    training_paths = {
        "non_dp_canary": runs
        / "canary_training/full_20260820/non_dp/run_summary.json",
        "dp_eps0p5_canary": runs
        / "canary_training/full_20260820_eps0p5/hooks_dp/run_summary.json",
        "dp_eps2_canary": runs
        / "canary_training/full_20260820/hooks_dp/run_summary.json",
        "dp_eps8_canary": runs
        / "canary_training/full_20260820_eps8/hooks_dp/run_summary.json",
        "stress_non_dp": runs
        / "canary_training/stress_full_20260820/non_dp/run_summary.json",
        "stress_dp_eps2": runs
        / "canary_training/stress_full_20260820/hooks_dp/run_summary.json",
    }
    training = {label: load_json(path) for label, path in training_paths.items()}

    lines = [
        "# 2026-08-20 실증적 Privacy 평가 결과",
        "",
        "> 실제 환자정보를 사용하지 않았다. Synthetic Canary의 환자 ID와 네 자리 코드는 모두 무작위 합성 데이터다.",
        "",
        "## 실험 조건",
        "",
        "- Model: VaultGemma-1B 4-bit NF4 + LoRA r=8, alpha=16",
        "- Canonical data: 전체 33,955개를 seed 42로 shuffle 후 8,000개, train/eval 7,200/800",
        "- DP: Poisson sampling, logical/physical batch 128/16, 342 steps, delta=1e-5, C=1, PRV",
        "- Canary: member 64개, non-member control 64개, 한 Canary당 한 privacy unit",
        "- Prefix-Suffix: short 10->20 token 각 196개, long 50->50 token member 128개",
        "- 모든 수치는 seed 42 단일 실행",
        "",
        "## 1. Prefix-Suffix 추출",
        "",
    ]
    legacy_stats = append_prefix_table(
        lines,
        "기존 head-split checkpoint",
        legacy_summaries,
        legacy_labels,
    )
    canonical_stats = append_prefix_table(
        lines,
        "Canonical shuffled split, Canary 교체 위치 64개 제외",
        canonical_summaries,
        canonical_labels,
    )

    legacy_indexes = {
        label: {
            (row["profile"], row["source_index"]): row
            for row in legacy_details[label]
        }
        for label in legacy_labels
    }
    example_keys = sorted(
        key
        for key, row in legacy_indexes["non_dp"].items()
        if key[0] == "qa_member_10_20"
        and row["exact_match"]
        and not legacy_indexes["dp_eps2_hooks"][key]["exact_match"]
    )[:3]
    lines.extend(
        [
            "### 정성 예시: 기존 non-DP exact, DP epsilon=2 non-exact",
            "",
            "| Source | Target | non-DP output | DP epsilon=2 output |",
            "|---:|---|---|---|",
        ]
    )
    for key in example_keys:
        non_dp_row = legacy_indexes["non_dp"][key]
        dp_row = legacy_indexes["dp_eps2_hooks"][key]
        lines.append(
            f"| {key[1]} | {markdown_cell(non_dp_row['target_suffix'])} | "
            f"{markdown_cell(non_dp_row['generated_suffix'])} | "
            f"{markdown_cell(dp_row['generated_suffix'])} |"
        )

    lines.extend(
        [
            "",
            "## 2. Canary 학습의 privacy-utility",
            "",
            "| 모델 | Target epsilon | Actual epsilon | Noise sigma | Eval loss | Eval PPL | 시간 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in canonical_labels[1:]:
        result = training[label]
        target = "-" if result["target_epsilon"] is None else f"{result['target_epsilon']:g}"
        actual = "-" if result["final_epsilon"] is None else f"{result['final_epsilon']:.4f}"
        sigma = "-" if result["noise_multiplier"] is None else f"{result['noise_multiplier']:.6f}"
        lines.append(
            f"| {label} | {target} | {actual} | {sigma} | "
            f"{result['eval']['example_mean_loss']:.4f} | "
            f"{result['eval']['example_mean_ppl']:.4f} | "
            f"{result['elapsed_training_sec'] / 60:.2f}분 |"
        )

    lines.extend(["", "## 3. Synthetic Canary 추출", ""])
    standard_canary_stats = append_canary_table(
        lines,
        "Standard v1: record 내부 반복 1·2·4·8회",
        standard_canary_summaries,
        standard_canary_details,
        standard_canary_labels,
    )
    stress_canary_stats = append_canary_table(
        lines,
        "Stress v2: record 내부 반복 4·8·16·32회",
        stress_canary_summaries,
        stress_canary_details,
        stress_canary_labels,
    )

    lines.extend(
        [
            "## 핵심 해석",
            "",
            "1. 기존 head-split non-DP는 short approximate member excess와 Fisher p=0.0277을 보였지만, shuffled canonical split에서는 재현되지 않았다.",
            "2. Canonical Prefix-Suffix는 모든 모델의 member/control 차이가 작고 long 50->50 exact는 전부 0건이다.",
            "3. Standard와 stress Canary 모두 Base/non-DP/DP에서 exact extraction 0건이며 score AUC는 무작위 0.5 부근이다.",
            "4. 따라서 이번 Canary 결과는 DP 우월성을 실증하지 못했다. 현재 recipe에서 단일-record 합성 코드는 non-DP도 검출 가능한 수준으로 암기하지 않았다는 negative result다.",
            "5. 반면 privacy-utility에서는 epsilon이 작아질수록 Eval loss가 증가하는 일관된 trade-off가 확인됐다.",
            "",
            "## 해석 제한과 다음 단계",
            "",
            "- 공격 실패는 DP의 증명이 아니며 formal epsilon·delta 보장과 함께 보고해야 한다.",
            "- Exposure는 공개한 128개 candidate code 안의 상대 rank로, 다른 candidate space와 직접 비교하지 않는다.",
            "- 최종 통계 주장을 위해 seed를 최소 3개로 늘려야 한다.",
            "- Canary 차이를 의도적으로 관찰하려면 중복 record stress 또는 더 긴 학습이 필요하지만, 중복 record는 group privacy 실험으로 별도 표기해야 한다.",
            "- 실제 개인정보나 환자 기록은 사용하지 않는다.",
            "",
        ]
    )

    combined = {
        "schema_version": 1,
        "legacy_prefix": {
            "summaries": legacy_summaries,
            "statistics": legacy_stats,
        },
        "canonical_prefix": {
            "summaries": canonical_summaries,
            "statistics": canonical_stats,
        },
        "training": training,
        "standard_canary": {
            "summaries": standard_canary_summaries,
            "statistics": standard_canary_stats,
        },
        "stress_canary": {
            "summaries": stress_canary_summaries,
            "statistics": stress_canary_stats,
        },
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
