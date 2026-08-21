#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compile_level1_patient_code import extraction_statistics, load_json


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


def method_dir(method: str) -> str:
    return f"{method}_eps2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    runs = args.results_root / "runs"
    method_root = runs / "method_comparison" / args.run_id
    method_runs: dict[str, dict[str, Any]] = {}
    statistics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        summary = load_json(method_root / method_dir(method) / "run_summary.json")
        if summary["run_type"] != "full":
            raise ValueError(f"Expected a full run for {method}")
        method_runs[method] = summary
        statistics[method] = extraction_statistics(summary["evaluation"])

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
    non_dp_statistics = extraction_statistics(non_dp_evaluation)

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
