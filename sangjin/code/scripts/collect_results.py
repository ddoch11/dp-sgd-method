#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ORDER = ["non_dp", "naive_dp", "hooks_dp", "vmap_dp", "expanded_weights_dp", "ghost_dp", "fastdp_bk"]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_full(date_root: Path) -> list[dict[str, Any]]:
    latest: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in (date_root / "runs").glob("**/run_summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("run_type") != "full":
            continue
        method = str(value.get("method"))
        current = latest.get(method)
        if current is None or path.stat().st_mtime > current[0].stat().st_mtime:
            latest[method] = (path, value)
    return [latest[name][1] for name in ORDER if name in latest]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    date_root = root / "experiments" / args.date
    items = load_full(date_root)
    first = items[0] if items else {}
    is_4bit = bool(first.get("load_in_4bit"))
    precision = "4-bit NF4" if is_4bit else "BF16"
    execution_note = (
        "4-bit run들은 여러 GPU에서 병렬·후속 배치로 실행했으므로 계산 지표는 참고값이다."
        if is_4bit
        else "신규 네 DP 방법은 GPU 4장 병렬 실행, BF16 Hooks는 GPU 0 단독, non-DP는 후속 실행이므로 계산 지표의 측정 조건을 함께 확인한다."
    )
    lines = [
        f"# {args.date} DP-SGD 방법 확장 비교",
        "",
        f"> VaultGemma-1B {precision}/eager attention 방법 비교다. {execution_note}",
        "",
        "## 공통 조건",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        "| Dataset | MedAlpaca flashcards, train/eval 7,200/800 |",
        f"| Model | VaultGemma-1B {precision}, eager attention |",
        "| LoRA | r8, alpha16, dropout0, 6,842,368 trainable |",
        "| Loss | response-only per-sequence response-token mean |",
        "| Sampling | Poisson, q=128/7200 |",
        "| Logical / physical batch | 128 / 8 |",
        "| Epoch / steps | 6 / 342 |",
        "| Privacy | target epsilon=2, delta=1e-5, C=1, sigma=1.015625 |",
        "| Accountant | 비교용 actual epsilon은 Opacus PRV |",
        "",
        "## Full 결과",
        "",
        "| 방법 | Actual ε | Eval loss | Eval PPL | 학습시간 | 처리량 | Torch peak VRAM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        evaluation = item.get("eval") or {}
        elapsed = item.get("elapsed_training_sec")
        lines.append(
            f"| {item.get('method')} | {fmt(item.get('final_epsilon'))} | "
            f"{fmt(evaluation.get('example_mean_loss'))} | {fmt(evaluation.get('example_mean_ppl'))} | "
            f"{fmt(elapsed / 60 if elapsed else None, 2)}분 | "
            f"{fmt(item.get('throughput_samples_per_sec'), 2)} samples/s | "
            f"{fmt(item.get('peak_vram_gb'), 2)}GB |"
        )
    lines.extend(["", "## 완료 상태", ""])
    for name in ORDER:
        if is_4bit and name == "vmap_dp":
            status = "excluded: bitsandbytes MatMul4Bit incompatible"
        else:
            status = "completed" if any(item.get("method") == name for item in items) else "pending"
        lines.append(f"- `{name}`: {status}")
    if any(item.get("method") == "fastdp_bk" for item in items):
        fastdp = next(item for item in items if item.get("method") == "fastdp_bk")
        lines.extend(
            [
                "",
                "## FastDP 내부 accountant",
                "",
                "```json",
                json.dumps(fastdp.get("fastdp_privacy"), indent=2, ensure_ascii=False),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 같은 수학을 구현한 방법은 Eval loss가 거의 같아야 한다. 차이가 크면 성능 우열보다 gradient scaling과 clipping 등가성을 먼저 점검한다.",
            f"- {execution_note}",
            "- 최종 효율 순위를 확정하려면 비교 방법을 동일 GPU에서 단독 재실행한다.",
            (
                "- 본 4-bit 표는 physical batch 8/eager attention 조건이며 기존 physical batch 16 정본과 계산 지표를 직접 합치지 않는다."
                if is_4bit
                else "- BF16 비교이므로 기존 4-bit Naive/Hooks의 VRAM·시간과 직접 비교하지 않는다."
            ),
            "- ε는 실제 개인정보 유출 확률이 아니라 accountant가 산출한 privacy loss 상한이다.",
            "",
        ]
    )
    output = date_root / "results.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
