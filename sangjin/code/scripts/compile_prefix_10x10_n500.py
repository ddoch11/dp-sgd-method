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


def profile_stats(summary: dict[str, Any]) -> dict[str, Any]:
    member = summary["profiles"]["qa_member_10_10"]
    control = summary["profiles"]["qa_nonmember_10_10"]
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
        "exact_member_excess": member_exact / member_total
        - control_exact / control_total,
        "exact_fisher_two_sided_p": fisher_exact(
            [
                [member_exact, member_total - member_exact],
                [control_exact, control_total - control_exact],
            ],
            alternative="two-sided",
        ).pvalue,
        "member_approximate": member_approx,
        "control_approximate": control_approx,
        "approximate_member_excess": member_approx / member_total
        - control_approx / control_total,
        "approximate_fisher_two_sided_p": fisher_exact(
            [
                [member_approx, member_total - member_approx],
                [control_approx, control_total - control_approx],
            ],
            alternative="two-sided",
        ).pvalue,
        "member_edit_similarity": float(member["mean_normalized_edit_similarity"]),
        "control_edit_similarity": float(control["mean_normalized_edit_similarity"]),
        "member_matching_prefix_tokens": float(member["mean_matching_prefix_tokens"]),
        "control_matching_prefix_tokens": float(control["mean_matching_prefix_tokens"]),
    }


def compact(value: Any, limit: int = 160) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    labels = [
        "base_n500",
        "non_dp_n500",
        "dp_eps0p5_n500",
        "dp_eps2_n500",
        "dp_eps8_n500",
    ]
    display = {
        "base_n500": "Base",
        "non_dp_n500": "non-DP LoRA",
        "dp_eps0p5_n500": "DP epsilon=0.5",
        "dp_eps2_n500": "DP epsilon=2",
        "dp_eps8_n500": "DP epsilon=8",
    }
    summaries = {
        label: load_json(args.run_dir / label / "summary.json") for label in labels
    }
    details = {
        label: load_jsonl(args.run_dir / label / "details.jsonl") for label in labels
    }
    member_hashes = {
        summary["profiles"]["qa_member_10_10"]["case_source_indices_sha256"]
        for summary in summaries.values()
    }
    control_hashes = {
        summary["profiles"]["qa_nonmember_10_10"]["case_source_indices_sha256"]
        for summary in summaries.values()
    }
    if len(member_hashes) != 1 or len(control_hashes) != 1:
        raise ValueError("Models were evaluated on different source-index sets")

    statistics = {label: profile_stats(summaries[label]) for label in labels}
    lines = [
        "# 2026-08-20 Canonical Prefix-Suffix 10->10, n=500",
        "",
        "> Canonical shuffled split에서 실제 학습 문장 Member 500개와 held-out Control 500개를 동일한 deterministic 공격으로 평가했다.",
        "",
        "## 실험 설정",
        "",
        "- Input: 전체 instruction/question + response 앞 10 token",
        "- Target: response 다음 10 token",
        "- Exact: 10 token 전체 일치",
        "- Approximate: token edit distance 1 이하",
        "- Decoding: greedy, do_sample=False, num_beams=1",
        "- Models: Base, non-DP LoRA, DP epsilon=0.5/2/8 Hooks",
        "- Synthetic Canary로 교체된 train 위치 64개 제외",
        "",
        "## 결과",
        "",
        "| 모델 | Member exact | Control exact | Exact excess | Exact p | Member approx | Control approx | Approx excess | Approx p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        stats = statistics[label]
        lines.append(
            f"| {display[label]} | {rate_text(stats['member_exact'], 500)} | "
            f"{rate_text(stats['control_exact'], 500)} | "
            f"{100 * stats['exact_member_excess']:+.2f}%p | "
            f"{stats['exact_fisher_two_sided_p']:.4f} | "
            f"{stats['member_approximate']}/500 | {stats['control_approximate']}/500 | "
            f"{100 * stats['approximate_member_excess']:+.2f}%p | "
            f"{stats['approximate_fisher_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| 모델 | Member edit similarity | Control edit similarity | Member matching-prefix | Control matching-prefix |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        stats = statistics[label]
        lines.append(
            f"| {display[label]} | {stats['member_edit_similarity']:.4f} | "
            f"{stats['control_edit_similarity']:.4f} | "
            f"{stats['member_matching_prefix_tokens']:.3f} | "
            f"{stats['control_matching_prefix_tokens']:.3f} |"
        )

    indexes = {
        label: {
            (row["profile"], int(row["source_index"])): row
            for row in details[label]
        }
        for label in labels
    }
    example_keys = sorted(
        key
        for key, row in indexes["non_dp_n500"].items()
        if key[0] == "qa_member_10_10"
        and row["exact_match"]
        and not indexes["base_n500"][key]["exact_match"]
        and not indexes["dp_eps2_n500"][key]["exact_match"]
    )[:3]
    lines.extend(
        [
            "",
            "## 정성 예시: non-DP exact, Base·DP epsilon=2 non-exact",
            "",
            "| Source | Target | Base | non-DP | DP epsilon=2 |",
            "|---:|---|---|---|---|",
        ]
    )
    for key in example_keys:
        lines.append(
            f"| {key[1]} | {compact(indexes['non_dp_n500'][key]['target_suffix'])} | "
            f"{compact(indexes['base_n500'][key]['generated_suffix'])} | "
            f"{compact(indexes['non_dp_n500'][key]['generated_suffix'])} | "
            f"{compact(indexes['dp_eps2_n500'][key]['generated_suffix'])} |"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- non-DP는 Base보다 Member와 Control 모두에서 exact·approximate continuation이 증가했다.",
            "- non-DP Member exact와 Control exact는 모두 79/500으로 같아 membership-dependent memorization excess는 없었다.",
            "- 모든 DP 모델도 Control이 Member와 같거나 더 높았으며 Fisher test에서 유의한 Member excess가 없었다.",
            "- 따라서 이 canonical 10->10 공격에서는 fine-tuning의 도메인 적응은 관찰됐지만, train 포함 여부에 따른 verbatim memorization은 탐지되지 않았다.",
            "- 결과는 seed 42 단일 checkpoint와 한 가지 greedy attack에 한정된다.",
            "",
        ]
    )

    combined = {
        "schema_version": 1,
        "member_source_indices_sha256": next(iter(member_hashes)),
        "control_source_indices_sha256": next(iter(control_hashes)),
        "summaries": summaries,
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
