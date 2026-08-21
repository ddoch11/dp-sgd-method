#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from compile_level1_patient_code import extraction_statistics, load_json, load_jsonl


METHODS = (
    "naive_dp",
    "hooks_dp",
    "vmap_dp",
    "expanded_weights_dp",
    "ghost_dp",
    "fastdp_bk",
)
DISPLAY_NAMES = {
    "naive_dp": "Naive Python loop",
    "hooks_dp": "Opacus Hooks",
    "vmap_dp": "Direct vmap",
    "expanded_weights_dp": "ExpandedWeights",
    "ghost_dp": "Ghost Clipping",
    "fastdp_bk": "FastDP Book-Keeping",
}
UTILITY_RUN_ID = "medalpaca_full_20260821"


def method_dir(method: str) -> str:
    return f"{method}_eps2"


def index_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {str(row["patient_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate patient_id in evaluation details")
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    runs = args.results_root / "runs"
    method_root = runs / "method_comparison" / args.run_id
    method_runs: dict[str, dict[str, Any]] = {}
    statistics: dict[str, dict[str, Any]] = {}
    method_details: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        run_root = method_root / method_dir(method)
        summary = load_json(run_root / "run_summary.json")
        if summary["run_type"] != "full":
            raise ValueError(f"Expected a full run for {method}")
        method_runs[method] = summary
        statistics[method] = extraction_statistics(summary["evaluation"])
        method_details[method] = load_jsonl(run_root / "evaluation/details.jsonl")

    non_dp_root = (
        runs
        / "training"
        / "pilot_grid_lr1e4_e40_20260821"
        / "non_dp"
    )
    non_dp_training = load_json(non_dp_root / "run_summary.json")
    non_dp_evaluation = load_json(
        non_dp_root / "checkpoints/epoch_040/evaluation/summary.json"
    )
    non_dp_details = load_jsonl(
        non_dp_root / "checkpoints/epoch_040/evaluation/details.jsonl"
    )
    non_dp_statistics = extraction_statistics(non_dp_evaluation)

    base_root = runs / "evaluation/base_20260821/base"
    base_details = load_jsonl(base_root / "details.jsonl")

    official_root = (
        runs
        / "training"
        / "tuned_dp_lr1e4_e40_20260821"
        / "opacus_hooks_dp_eps2"
    )
    official_training = load_json(official_root / "run_summary.json")
    official_evaluation = load_json(
        official_root / "checkpoints/epoch_040/evaluation/summary.json"
    )
    official_statistics = extraction_statistics(official_evaluation)

    utility_root = runs / "utility" / UTILITY_RUN_ID
    utility_evaluations = {
        label: load_json(utility_root / f"{label}.json")
        for label in ("base", "non_dp", *METHODS)
    }
    for label, utility in utility_evaluations.items():
        if utility["run_type"] != "full" or utility["evaluated_samples"] != 800:
            raise ValueError(f"Incomplete MedAlpaca utility evaluation for {label}")
    base_utility_loss = utility_evaluations["base"]["metrics"][
        "example_mean_loss"
    ]

    all_details = {
        "base": base_details,
        "non_dp": non_dp_details,
        **method_details,
    }
    detail_indexes = {
        label: index_details(rows) for label, rows in all_details.items()
    }
    patient_ids = sorted(detail_indexes["base"])
    if len(patient_ids) != 1000:
        raise ValueError("Expected exactly 1,000 evaluated patients")
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

    output_analysis = {
        label: member_output_analysis(rows) for label, rows in all_details.items()
    }
    example_ids = [
        patient_id
        for patient_id in patient_ids
        if detail_indexes["non_dp"][patient_id]["membership"] == "member"
        and detail_indexes["non_dp"][patient_id]["exact_extraction"]
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

    lines = [
        "# 2026-08-21 Level 1 합성 환자 코드 DP backend 비교",
        "",
        "> 합성 환자 코드 memorization task를 여섯 DP-SGD per-sample gradient backend로 각각 원본 VaultGemma-1B에서 새로 학습했다. 모든 DP run의 목표 privacy는 epsilon=2, delta=1e-5다.",
        "",
        "## 목적",
        "",
        "DP-SGD의 clipping, Gaussian noise, privacy accounting을 유지하면서 per-sample gradient 계산 backend만 바꿨을 때 utility와 계산 특성이 일치하는지 확인한다.",
        "",
        "## 공통 조건",
        "",
        "- VaultGemma-1B BF16 + 새 LoRA r=8, alpha=16, dropout=0",
        "- 합성 Member 500개 학습, 합성 Control 500개 미학습",
        "- 40 epoch, 640 optimizer steps, lr=1e-4, weight decay=0",
        "- Logical batch 32, physical batch 16, Naive만 physical batch 1",
        "- Bernoulli Poisson sampling q=0.064, sampling seed 20042",
        "- epsilon=2, delta=1e-5, C=1, PRV accountant, sigma=3.37890625",
        "- 다섯 manual backend는 noise seed 10042까지 동일",
        "- FastDP는 동일 sigma와 외부 PRV accountant를 사용하고 내부 RDP 값도 별도 기록",
        "",
        "## 기준선",
        "",
        f"- non-DP 40 epoch: Member exact {non_dp_statistics['member_exact']}/500, Control exact {non_dp_statistics['control_exact']}/500, Score AUC {non_dp_statistics['target_score_membership_auc']:.4f}",
        f"- 기존 Opacus PrivacyEngine Hooks: Actual epsilon {official_training['final_epsilon']:.4f}, Member exact {official_statistics['member_exact']}/500, Control exact {official_statistics['control_exact']}/500, Score AUC {official_statistics['target_score_membership_auc']:.4f}",
        "- 기존 PrivacyEngine run은 DataLoader가 만든 유효 sample rate로 accounting하므로 공통 manual 하네스와 Actual epsilon이 다르다. 아래 주 비교표에는 섞지 않는다.",
        "",
        "## 최종 결과",
        "",
        "| 방법 | Actual epsilon | sigma | Final train loss | Member exact | Control exact | Score AUC | 시간 | 처리량 | Peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        run = method_runs[method]
        stats = statistics[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | {run['final_epsilon']:.4f} | "
            f"{run['noise_multiplier']:.6f} | {run['loss_last']:.4f} | "
            f"{stats['member_exact']}/500 | {stats['control_exact']}/500 | "
            f"{stats['target_score_membership_auc']:.4f} | "
            f"{run['elapsed_training_sec'] / 60:.2f}분 | "
            f"{run['throughput_samples_per_sec']:.2f} samples/s | "
            f"{run['peak_vram_gb']:.2f}GB |"
        )

    lines.extend(
        [
            "",
            "## MedAlpaca utility와 forgetting",
            "",
            "Level 1 모델은 MedAlpaca로 학습한 모델이 아니다. 기존 BF16 비교와 같은 `medalpaca/medical_meadow_medical_flashcards` 앞 8,000개 중 고정 eval 800개를 사용해, 합성 code fine-tuning 후 기존 의료 QA response loss가 얼마나 변했는지 측정했다.",
            "",
            "| 모델 | Eval loss | Eval PPL | Base 대비 delta loss |",
            "|---|---:|---:|---:|",
        ]
    )
    utility_display = {"base": "Base", "non_dp": "non-DP LoRA", **DISPLAY_NAMES}
    for label in ("base", "non_dp", *METHODS):
        metrics = utility_evaluations[label]["metrics"]
        delta_loss = metrics["example_mean_loss"] - base_utility_loss
        lines.append(
            f"| {utility_display[label]} | {metrics['example_mean_loss']:.4f} | "
            f"{metrics['example_mean_ppl']:.4f} | {delta_loss:+.4f} |"
        )
    lines.extend(
        [
            "",
            "- Base는 별도 fine-tuning이 없는 기준이다.",
            "- non-DP는 synthetic Member mapping을 강하게 암기했지만 MedAlpaca Eval loss가 크게 증가해 catastrophic forgetting 신호를 보였다.",
            "- epsilon=2 DP backend들은 synthetic mapping을 복원하지 못한 대신 MedAlpaca loss 증가는 상대적으로 작았다.",
            "- 이 평가는 teacher-forcing response-only loss/PPL이며 정답 accuracy가 아니다.",
            "- 과거 MedAlpaca train 7,200개로 직접 fine-tuning한 Eval loss 약 1.21과는 학습 task가 다르므로 직접 성능 순위를 비교하지 않는다.",
        ]
    )

    lines.extend(
        [
            "",
            "## 실제 생성 출력 확인",
            "",
            "평가는 학습 prompt와 같은 `Return only the private code` 형식을 사용했다. 추가 공격 instruction은 넣지 않았고 `do_sample=False`, `num_beams=1`, `max_new_tokens=8`로 1,000개 전부 생성했다.",
            "",
            "### Member 출력 분포",
            "",
            "| 모델 | 고유 output 수 | 네 자리 code 출력 | Target exact | 최빈 output | 빈도 |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    output_display = {
        "base": "Base",
        "non_dp": "non-DP LoRA",
        **DISPLAY_NAMES,
    }
    for label in ("base", "non_dp", *METHODS):
        analysis = output_analysis[label]
        top = analysis["top_outputs"][0]
        lines.append(
            f"| {output_display[label]} | {analysis['unique_generated_outputs']} | "
            f"{analysis['outputs_with_four_digit_code']}/500 | "
            f"{analysis['exact_extractions']}/500 | {md_code(top['output'])} | "
            f"{top['count']}/500 |"
        )

    lines.extend(
        [
            "",
            "### 대표 Member 예시 1",
            "",
            "| Patient | Target | Base | non-DP | Naive | Hooks | Direct vmap |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for example in qualitative_examples:
        outputs = example["outputs"]
        lines.append(
            f"| {example['patient_id']} | {example['target']} | "
            f"{md_code(outputs['base']['generated_text'])} | "
            f"{md_code(outputs['non_dp']['generated_text'])} | "
            f"{md_code(outputs['naive_dp']['generated_text'])} | "
            f"{md_code(outputs['hooks_dp']['generated_text'])} | "
            f"{md_code(outputs['vmap_dp']['generated_text'])} |"
        )

    lines.extend(
        [
            "",
            "### 대표 Member 예시 2",
            "",
            "| Patient | Target | ExpandedWeights | Ghost | FastDP BK |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for example in qualitative_examples:
        outputs = example["outputs"]
        lines.append(
            f"| {example['patient_id']} | {example['target']} | "
            f"{md_code(outputs['expanded_weights_dp']['generated_text'])} | "
            f"{md_code(outputs['ghost_dp']['generated_text'])} | "
            f"{md_code(outputs['fastdp_bk']['generated_text'])} |"
        )

    lines.extend(
        [
            "",
            "DP 모델은 네 자리 형식 자체는 대부분 생성했지만 `1000`, `1100`, `1111`, `2000` 같은 소수 output으로 집중됐고 실제 target과 일치한 경우는 없었다. 전체 1,000개 원문 output과 exact 판정은 CSV에 저장했다.",
        ]
    )

    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 여섯 방법은 같은 DP-SGD update를 서로 다른 계산 경로로 구현한 것이므로 privacy와 utility가 비슷한 것이 정상이다.",
            "- Member/Control exact와 Score AUC는 backend가 empirical memorization 결론을 바꾸는지 확인하는 지표다.",
            "- Naive는 샘플마다 backward를 실행해 가장 느리지만 per-sample gradient 기준선 역할을 한다.",
            "- Hooks, Direct vmap, ExpandedWeights는 per-sample gradient를 batch 단위로 실체화하는 서로 다른 backend다.",
            "- Ghost와 FastDP는 전체 per-sample gradient 텐서를 만들지 않는 clipping 계열이다.",
            "- 모든 run은 원본 base에서 독립적으로 시작했으며 이전 adapter를 이어서 학습하지 않았다.",
            "- 시간·처리량은 네 GPU 병렬 실행이 섞인 compatibility 측정치다. 최종 효율 순위를 주장하려면 각 방법의 단독 재실행이 필요하다.",
            "- 합성 key-value memorization stress와 단일 seed 결과이며 일반 의료 QA utility나 모든 privacy 공격을 대표하지 않는다.",
            "",
        ]
    )

    combined = {
        "schema_version": 1,
        "report_variant": "level1_patient_code_dp_backend_eps2",
        "run_id": args.run_id,
        "manifest_payload_sha256": next(iter(method_runs.values()))[
            "manifest_payload_sha256"
        ],
        "non_dp_reference": {
            "training": non_dp_training,
            "evaluation": non_dp_evaluation,
            "statistics": non_dp_statistics,
        },
        "official_opacus_reference": {
            "training": official_training,
            "evaluation": official_evaluation,
            "statistics": official_statistics,
        },
        "method_runs": method_runs,
        "statistics": statistics,
        "medalpaca_utility": utility_evaluations,
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
    csv_labels = ("base", "non_dp", *METHODS)
    csv_fieldnames = ["patient_id", "membership", "target"]
    for label in csv_labels:
        csv_fieldnames.extend(
            [f"{label}_output", f"{label}_extracted_code", f"{label}_exact"]
        )
    with args.output_csv.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=csv_fieldnames, lineterminator="\n"
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
