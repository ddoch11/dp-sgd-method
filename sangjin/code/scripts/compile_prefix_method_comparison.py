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
        f"CI {100 * lower:.2f}-{100 * upper:.2f}%)"
    )


def statistics(summary: dict[str, Any]) -> dict[str, Any]:
    member = summary["profiles"]["qa_member_10_20"]
    control = summary["profiles"]["qa_nonmember_10_20"]
    long_result = summary["profiles"]["vaultgemma_member_50_50"]
    member_total = int(member["samples"])
    control_total = int(control["samples"])
    member_exact = int(member["exact_matches"])
    control_exact = int(control["exact_matches"])
    member_approx = int(member["approximate_matches_10pct"])
    control_approx = int(control["approximate_matches_10pct"])
    return {
        "member_total": member_total,
        "control_total": control_total,
        "member_exact": member_exact,
        "control_exact": control_exact,
        "exact_excess": member_exact / member_total - control_exact / control_total,
        "exact_fisher_p": fisher_exact(
            [
                [member_exact, member_total - member_exact],
                [control_exact, control_total - control_exact],
            ],
            alternative="two-sided",
        ).pvalue,
        "member_approximate": member_approx,
        "control_approximate": control_approx,
        "approximate_excess": member_approx / member_total
        - control_approx / control_total,
        "approximate_fisher_p": fisher_exact(
            [
                [member_approx, member_total - member_approx],
                [control_approx, control_total - control_approx],
            ],
            alternative="two-sided",
        ).pvalue,
        "member_edit_similarity": float(member["mean_normalized_edit_similarity"]),
        "control_edit_similarity": float(control["mean_normalized_edit_similarity"]),
        "long_exact": int(long_result["exact_matches"]),
        "long_approximate": int(long_result["approximate_matches_10pct"]),
        "long_samples": int(long_result["samples"]),
        "long_edit_similarity": float(long_result["mean_normalized_edit_similarity"]),
    }


def detail_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["profile"], int(row["source_index"])): row for row in rows}


def output_agreement(
    reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reference = detail_index(reference_rows)
    candidate = detail_index(candidate_rows)
    if set(reference) != set(candidate):
        raise ValueError("Method comparison uses different Prefix-Suffix cases")
    matches = sum(
        reference[key]["generated_token_ids"] == candidate[key]["generated_token_ids"]
        for key in reference
    )
    return {
        "identical_outputs": matches,
        "total_outputs": len(reference),
        "identical_output_rate": matches / len(reference),
    }


def append_group(
    lines: list[str],
    title: str,
    summaries: dict[str, Any],
    details: dict[str, list[dict[str, Any]]],
    labels: list[str],
    display_names: dict[str, str],
    hooks_label: str,
) -> dict[str, Any]:
    lines.extend(
        [
            f"## {title}",
            "",
            "| 모델 | Member exact | Control exact | Exact excess | Exact p | Member approx | Control approx | Approx p | Long exact |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    result: dict[str, Any] = {"models": {}, "agreement_with_hooks": {}}
    for label in labels:
        stats = statistics(summaries[label])
        result["models"][label] = stats
        lines.append(
            f"| {display_names[label]} | "
            f"{rate_text(stats['member_exact'], stats['member_total'])} | "
            f"{rate_text(stats['control_exact'], stats['control_total'])} | "
            f"{100 * stats['exact_excess']:+.2f}%p | {stats['exact_fisher_p']:.4f} | "
            f"{stats['member_approximate']}/{stats['member_total']} | "
            f"{stats['control_approximate']}/{stats['control_total']} | "
            f"{stats['approximate_fisher_p']:.4f} | "
            f"{stats['long_exact']}/{stats['long_samples']} |"
        )
    lines.extend(
        [
            "",
            "### DP checkpoint output agreement with Hooks",
            "",
            "| 방법 | 완전히 같은 generation | 비율 | Member edit similarity | Long edit similarity |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        if label in {labels[0], labels[1]}:
            continue
        agreement = output_agreement(details[hooks_label], details[label])
        result["agreement_with_hooks"][label] = agreement
        stats = result["models"][label]
        lines.append(
            f"| {display_names[label]} | {agreement['identical_outputs']}/{agreement['total_outputs']} | "
            f"{100 * agreement['identical_output_rate']:.2f}% | "
            f"{stats['member_edit_similarity']:.4f} | {stats['long_edit_similarity']:.4f} |"
        )
    lines.append("")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    runs = args.results_root / "runs" / "prefix_suffix"
    groups = {
        "4bit": {
            "title": "4-bit NF4 checkpoint 비교",
            "run_dir": runs / "existing_4bit_20260820_n196",
            "labels": [
                "base",
                "non_dp",
                "naive_dp_4bit",
                "dp_eps2_hooks",
                "expanded_weights_dp_4bit",
                "ghost_dp_4bit",
                "fastdp_bk_4bit",
            ],
            "display": {
                "base": "Base",
                "non_dp": "non-DP",
                "naive_dp_4bit": "Naive DP",
                "dp_eps2_hooks": "Opacus Hooks",
                "expanded_weights_dp_4bit": "ExpandedWeights",
                "ghost_dp_4bit": "Ghost Clipping",
                "fastdp_bk_4bit": "FastDP BK",
            },
            "hooks": "dp_eps2_hooks",
        },
        "bf16": {
            "title": "BF16 checkpoint 비교",
            "run_dir": runs / "methods_bf16_20260820",
            "labels": [
                "base_bf16",
                "non_dp_bf16",
                "naive_dp_bf16",
                "hooks_dp_bf16",
                "direct_vmap_bf16",
                "expanded_weights_bf16",
                "ghost_dp_bf16",
                "fastdp_bk_bf16",
            ],
            "display": {
                "base_bf16": "Base",
                "non_dp_bf16": "non-DP",
                "naive_dp_bf16": "Naive DP",
                "hooks_dp_bf16": "Opacus Hooks",
                "direct_vmap_bf16": "Direct vmap",
                "expanded_weights_bf16": "ExpandedWeights",
                "ghost_dp_bf16": "Ghost Clipping",
                "fastdp_bk_bf16": "FastDP BK",
            },
            "hooks": "hooks_dp_bf16",
        },
    }
    combined: dict[str, Any] = {"schema_version": 1, "groups": {}}
    lines = [
        "# 2026-08-20 Prefix-Suffix 방법별 비교",
        "",
        "> 기존 full checkpoint를 대상으로 동일한 deterministic Prefix-Suffix 공격을 수행했다. 4-bit와 BF16은 별도 표로 비교한다.",
        "",
        "- Short: member/non-member 각 196개, response prefix 10 token -> suffix 20 token",
        "- Long: member 128개, response prefix 50 token -> suffix 50 token",
        "- DP checkpoint: target epsilon=2, actual epsilon 약 1.9998",
        "- Dataset: 각 checkpoint가 실제 학습한 원본 앞 8,000개 head split",
        "",
    ]
    for group_name, definition in groups.items():
        summaries = {
            label: load_json(definition["run_dir"] / label / "summary.json")
            for label in definition["labels"]
        }
        details = {
            label: load_jsonl(definition["run_dir"] / label / "details.jsonl")
            for label in definition["labels"]
        }
        group_result = append_group(
            lines,
            definition["title"],
            summaries,
            details,
            definition["labels"],
            definition["display"],
            definition["hooks"],
        )
        group_result["summaries"] = summaries
        combined["groups"][group_name] = group_result

    lines.extend(
        [
            "## 해석",
            "",
            "- 4-bit의 다섯 DP 방법은 short exact 5/196 대 6/196, approximate 8/196 대 8/196으로 동일했다.",
            "- BF16의 여섯 DP 방법은 short exact 6/196 대 6/196으로 동일했고 approximate도 Ghost의 member 10건을 제외하면 11/196 대 8/196으로 같았다.",
            "- Long 50->50 exact와 approximate는 모든 모델에서 0건이었다.",
            "- 따라서 per-example gradient 계산 backend는 이 실험의 privacy extraction 결과를 바꾸지 않았다.",
            "- non-DP member excess는 4-bit와 BF16에서 모두 나타났지만 head split 결과이므로 canonical shuffled split 결과와 함께 제한적으로 해석해야 한다.",
            "- Direct vmap은 BF16 checkpoint만 존재하며 4-bit 표에는 포함하지 않았다.",
            "",
        ]
    )

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
