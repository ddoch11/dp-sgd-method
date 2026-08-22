#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.stats import fisher_exact


DISPLAY_NAMES = {
    "base": "Base",
    "non_dp": "non-DP LoRA",
    "naive_dp": "Naive DP-SGD",
    "hooks_dp": "Opacus Hooks",
    "vmap_dp": "Direct vmap",
    "expanded_weights_dp": "ExpandedWeights",
    "ghost_dp": "Ghost Clipping",
    "fastdp_bk": "FastDP Book-Keeping",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def find_single_summary(
    experiment_root: Path, method: str, epsilon_slug: str
) -> dict[str, Any]:
    matches = sorted(
        (experiment_root / "runs" / method / epsilon_slug).glob(
            "*/run_summary.json"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one summary for {method}/{epsilon_slug}, got {matches}"
        )
    return load_json(matches[0])


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
        "member_control_exact_excess": (
            member_exact / member_total - control_exact / control_total
        ),
        "fisher_two_sided_p": fisher_exact(
            [
                [member_exact, member_total - member_exact],
                [control_exact, control_total - control_exact],
            ],
            alternative="two-sided",
        ).pvalue,
        "target_score_membership_auc": summary["target_score_membership_auc"],
    }


def index_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["patient_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate patient_id in details")
    return indexed


def member_output_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    members = [row for row in rows if row["membership"] == "member"]
    counts = Counter(str(row["generated_text"]) for row in members)
    return {
        "samples": len(members),
        "unique_generated_outputs": len(counts),
        "outputs_with_four_digit_code": sum(
            row["extracted_code"] is not None for row in members
        ),
        "exact_extractions": sum(bool(row["exact_extraction"]) for row in members),
        "top_outputs": [
            {"output": output, "count": count}
            for output, count in counts.most_common(5)
        ],
    }


def md_code(value: str) -> str:
    escaped = value.replace("|", "\\|").replace("`", "\\`")
    escaped = escaped.replace("\r", " ").replace("\n", " ")
    return f"`{escaped}`"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--extraction-run-id", required=True)
    parser.add_argument("--base-utility", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["non_dp", "hooks_dp", "expanded_weights_dp", "vmap_dp"],
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    experiment_root = args.results_root / "experiments" / args.experiment_id
    training: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        epsilon_slug = "none" if method == "non_dp" else "eps2"
        training[method] = find_single_summary(
            experiment_root, method, epsilon_slug
        )

    extraction_root = (
        args.results_root
        / "experiments"
        / "evaluation"
        / args.extraction_run_id
    )
    extraction_summaries: dict[str, dict[str, Any]] = {}
    extraction_details: dict[str, list[dict[str, Any]]] = {}
    for label in ("base", *args.methods):
        extraction_summaries[label] = load_json(
            extraction_root / label / "summary.json"
        )
        extraction_details[label] = load_jsonl(
            extraction_root / label / "details.jsonl"
        )

    statistics = {
        label: extraction_statistics(summary)
        for label, summary in extraction_summaries.items()
    }
    output_analysis = {
        label: member_output_analysis(rows)
        for label, rows in extraction_details.items()
    }
    utility = {"base": load_json(args.base_utility)}
    utility.update({method: run["eval"] for method, run in training.items()})

    detail_indexes = {
        label: index_details(rows) for label, rows in extraction_details.items()
    }
    patient_ids = sorted(detail_indexes["base"])
    if len(patient_ids) != 1000:
        raise ValueError("Expected 1,000 synthetic Member/Control records")
    for patient_id in patient_ids:
        reference = detail_indexes["base"][patient_id]
        for label, indexed in detail_indexes.items():
            row = indexed.get(patient_id)
            if row is None:
                raise ValueError(f"Missing {patient_id} in {label}")
            if (
                row["private_code"] != reference["private_code"]
                or row["membership"] != reference["membership"]
            ):
                raise ValueError(f"Mismatched record for {patient_id} in {label}")

    positive_label = "non_dp"
    example_ids = [
        patient_id
        for patient_id in patient_ids
        if detail_indexes[positive_label][patient_id]["membership"] == "member"
        and detail_indexes[positive_label][patient_id]["exact_extraction"]
    ][:8]
    qualitative_examples = [
        {
            "patient_id": patient_id,
            "target": detail_indexes["base"][patient_id]["private_code"],
            "outputs": {
                label: {
                    "generated_text": indexed[patient_id]["generated_text"],
                    "extracted_code": indexed[patient_id]["extracted_code"],
                    "exact_extraction": indexed[patient_id]["exact_extraction"],
                }
                for label, indexed in detail_indexes.items()
            },
        }
        for patient_id in example_ids
    ]

    representative_dp = next(
        run for method, run in training.items() if method != "non_dp"
    )
    vectorized_run = training.get("hooks_dp", representative_dp)
    lines = [
        "# 2026-08-21 Mixed Private MedAlpaca, 30 epoch",
        "",
        "> MedAlpaca train 7,200개와 synthetic private Member 500개를 함께 fine-tuning한 과제 정본 실험이다. Synthetic Control 500개와 MedAlpaca eval 800개는 학습에 사용하지 않았다.",
        "",
        "## 데이터와 설정",
        "",
        "| 구분 | 수 | 학습 포함 |",
        "|---|---:|---|",
        "| MedAlpaca train | 7,200 | 포함 |",
        "| Synthetic private Member | 500 | 포함 |",
        "| MedAlpaca eval | 800 | 미포함 |",
        "| Synthetic Control | 500 | 미포함 |",
        "",
        f"- 총 train 7,700개, 30 epoch, {representative_dp['planned_steps']} optimizer steps",
        f"- Logical batch {representative_dp['logical_batch_size']}, physical batch {vectorized_run['physical_batch_size']} (Naive만 1)",
        f"- Poisson q={representative_dp['sample_rate']:.8f}",
        f"- epsilon=2, delta=1e-5, C=1, sigma={representative_dp['noise_multiplier']}",
        "- VaultGemma-1B BF16 + LoRA r8/alpha16/dropout0",
        "",
        "## Privacy와 utility 결과",
        "",
        "| 모델 | Actual epsilon | MedAlpaca Eval loss | Eval PPL | Member exact | Control exact | Score AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    base_metrics = utility["base"]["metrics"]
    base_stats = statistics["base"]
    lines.append(
        f"| Base | - | {base_metrics['example_mean_loss']:.4f} | "
        f"{base_metrics['example_mean_ppl']:.4f} | "
        f"{base_stats['member_exact']}/500 | {base_stats['control_exact']}/500 | "
        f"{base_stats['target_score_membership_auc']:.4f} |"
    )
    for method in args.methods:
        run = training[method]
        metrics = utility[method]
        stats = statistics[method]
        epsilon_text = "-" if method == "non_dp" else f"{run['final_epsilon']:.4f}"
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {epsilon_text} | "
            f"{metrics['example_mean_loss']:.4f} | {metrics['example_mean_ppl']:.4f} | "
            f"{stats['member_exact']}/500 | {stats['control_exact']}/500 | "
            f"{stats['target_score_membership_auc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 계산 결과",
            "",
            "| 방법 | Final train loss | 시간 | 처리량 | Peak VRAM | Naive 대비 속도 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    naive_time = (
        training["naive_dp"]["elapsed_training_sec"]
        if "naive_dp" in training
        else None
    )
    for method in args.methods:
        run = training[method]
        if naive_time is None or method == "non_dp":
            speedup_text = "-"
        else:
            speedup_text = f"{naive_time / run['elapsed_training_sec']:.2f}x"
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {run['loss_last']:.4f} | "
            f"{run['elapsed_training_sec'] / 60:.2f}분 | "
            f"{run['throughput_samples_per_sec']:.2f}/s | "
            f"{run['peak_vram_gb']:.2f}GB | {speedup_text} |"
        )

    lines.extend(
        [
            "",
            "## Member 실제 output 분포",
            "",
            "| 모델 | 고유 output | 네 자리 code | Target exact | 최빈 output | 빈도 |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for label in ("base", *args.methods):
        analysis = output_analysis[label]
        top = analysis["top_outputs"][0]
        lines.append(
            f"| {DISPLAY_NAMES[label]} | {analysis['unique_generated_outputs']} | "
            f"{analysis['outputs_with_four_digit_code']}/500 | "
            f"{analysis['exact_extractions']}/500 | {md_code(top['output'])} | "
            f"{top['count']}/500 |"
        )

    if qualitative_examples:
        table_labels = ("base", *args.methods)
        header = "| Patient | Target | " + " | ".join(
            DISPLAY_NAMES[label] for label in table_labels
        ) + " |"
        separator = "|---|---:|" + "---|" * len(table_labels)
        lines.extend(["", "## 대표 Member 실제 output", "", header, separator])
        for example in qualitative_examples:
            outputs = example["outputs"]
            values = " | ".join(
                md_code(outputs[label]["generated_text"])
                for label in table_labels
            )
            lines.append(
                f"| {example['patient_id']} | {example['target']} | {values} |"
            )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 이 결과는 MedAlpaca와 private synthetic record를 실제로 함께 fine-tuning한 조건이다.",
            "- Member/Control extraction과 MedAlpaca held-out utility를 함께 보고한다.",
            "- epsilon은 관측된 유출 확률이 아니라 sample-level DP privacy loss 상한이다.",
            "- 시간·처리량은 각 방법이 전용 GPU에서 실행된 측정값이지만 일부 run은 서버 내 병렬 실행이므로 최종 순위에는 이 조건을 함께 표기한다.",
            "- 단일 seed와 실험용 비보안 RNG 결과이며 최종 통계 주장 전 seed 반복이 필요하다.",
            "- Synthetic-only standalone Level 1 결과는 최종 과제 결과에서 제외한다.",
            "",
        ]
    )

    combined = {
        "schema_version": 1,
        "report_variant": "mixed_private_medalpaca_e30",
        "experiment_id": args.experiment_id,
        "extraction_run_id": args.extraction_run_id,
        "training": training,
        "utility": utility,
        "extraction_summaries": extraction_summaries,
        "statistics": statistics,
        "output_analysis": output_analysis,
        "qualitative_examples": qualitative_examples,
        "full_output_csv": args.output_csv.name,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(combined, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    with args.output_md.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))

    csv_labels = ("base", *args.methods)
    fieldnames = ["patient_id", "membership", "target"]
    for label in csv_labels:
        fieldnames.extend(
            [f"{label}_output", f"{label}_extracted_code", f"{label}_exact"]
        )
    with args.output_csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for patient_id in patient_ids:
            reference = detail_indexes["base"][patient_id]
            row: dict[str, Any] = {
                "patient_id": patient_id,
                "membership": reference["membership"],
                "target": reference["private_code"],
            }
            for label in csv_labels:
                detail = detail_indexes[label][patient_id]
                row[f"{label}_output"] = detail["generated_text"]
                row[f"{label}_extracted_code"] = detail["extracted_code"]
                row[f"{label}_exact"] = detail["exact_extraction"]
            writer.writerow(row)
    print(f"created {args.output_json}, {args.output_md}, and {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
