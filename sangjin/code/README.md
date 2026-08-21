# DP-SGD 방법 확장 비교

기존 `1차년도_final`의 Naive/Opacus Hooks 결과를 보존하고, 나머지 계산 방법을 검증하는 별도 실험 폴더다.

## 비교 대상

- `hooks_dp`: Opacus `GradSampleModule` hooks, BF16 공정 비교 기준
- `vmap_dp`: 모델 전체 `torch.func.vmap(grad_and_value)` 직접 적용
- `expanded_weights_dp`: Opacus `GradSampleModuleExpandedWeights`
- `ghost_dp`: Opacus Ghost Clipping 2-pass
- `fastdp_bk`: FastDP Book-Keeping base `ghost` mode, 별도 스크립트와 환경

## 공통 조건

- VaultGemma-1B BF16, eager attention
- LoRA r=8, alpha=16, dropout=0
- MedAlpaca flashcards 8,000개, train/eval 7,200/800
- response-only per-sequence loss
- Poisson expected logical batch 128
- physical batch 8
- 6 epochs, 342 optimizer steps
- ε=2, δ=1e-5, C=1, PRV accountant

직접 `vmap+grad`는 bitsandbytes `MatMul4Bit`이 `torch.func`에 필요한 `setup_context`를 구현하지 않아 4-bit에서 실행되지 않는다. 따라서 이 폴더는 네 방법을 BF16 공통 조건으로 맞춘다. 기존 4-bit 결과와 시간·VRAM을 직접 합쳐 비교하지 않는다.

## 준비

FastDP 구현은 고정된 외부 저장소를 Git submodule로 사용한다. 저장소를 받은 뒤 다음 명령을 먼저 실행한다.

```bash
git submodule update --init --recursive
```

## 실행

```bash
./scripts/run_one.sh vmap_dp 2 0 2026-08-19 2
./scripts/run_one.sh expanded_weights_dp 2 1 2026-08-19 2
./scripts/run_one.sh ghost_dp 2 2 2026-08-19 2
```

`max_steps=2` smoke 성공 후 `max_steps=0`으로 full run한다.

## 실증적 Privacy 평가 코드

| 파일 | 역할 |
|---|---|
| `scripts/evaluate_prefix_suffix.py` | 기존 train/non-member 문장의 deterministic continuation 추출 |
| `scripts/create_synthetic_canary_manifest.py` | 합성 member/control Canary와 dataset hash 생성 |
| `scripts/train_synthetic_canary.py` | 동일 Canary 데이터로 non-DP 또는 DP epsilon=2 재학습 |
| `scripts/evaluate_synthetic_canary.py` | open/guided extraction, candidate rank, exposure 측정 |
| `scripts/compile_privacy_evaluation.py` | raw 결과를 JSON·Markdown 최종 보고서로 통합 |
| `scripts/compile_prefix_method_comparison.py` | 4-bit·BF16 방법별 Prefix-Suffix 결과 통합 |
| `scripts/compile_prefix_10x10_n500.py` | Canonical 10→10 Member/Control 500개 결과 통합 |
| `scripts/run_post_training_privacy_evals.sh` | 학습 종료 후 Canary와 Prefix-Suffix 평가를 순차 실행 |
| `src/privacy_eval_common.py` | 데이터 selection, response loss, model loading 공통 코드 |

### Level 1 합성 환자 코드

| 파일 | 역할 |
|---|---|
| `scripts/create_level1_patient_codes.py` | Member/Control 500개씩 고유 alias-code manifest 생성 |
| `scripts/train_level1_patient_codes.py` | BF16 non-DP 또는 Opacus Hooks DP-SGD 학습 |
| `scripts/evaluate_level1_patient_codes.py` | code exact extraction과 target-score AUC 평가 |
| `scripts/compile_level1_patient_code.py` | pilot·DP sweep·정성 예시 결과 통합 |
| `scripts/train_level1_patient_code_methods.py` | Level 1 task의 Naive/Hooks/vmap/EW/Ghost/FastDP epsilon=2 공통 학습 |
| `scripts/compile_level1_patient_code_methods.py` | Level 1 DP backend 비교 결과 통합 |
| `scripts/evaluate_level1_medalpaca_utility.py` | Level 1 checkpoint의 고정 MedAlpaca eval 800개 response-only loss/PPL 평가 |
| `src/level1_patient_code_common.py` | prompt, tokenizer, model, evaluation 공통 코드 |

실험 설정은 `../configs/privacy_evaluation.yaml`에 있다. 대용량 adapter와 raw run은 `../results/privacy_eval/runs/`에 저장되며 Git에서 제외된다. Canary manifest와 최종 요약만 저장소에서 관리한다.

```bash
python scripts/create_synthetic_canary_manifest.py \
  --config ../configs/privacy_evaluation.yaml \
  --output ../results/privacy_eval/synthetic_canary_manifest.json
```

Canary는 기존 7,200개 train record 중 64개를 교체하므로 데이터 크기와 sampling rate `128/7200`을 유지한다. Standard v1은 한 record 내부에서 1·2·4·8회, stress v2는 4·8·16·32회 반복한다. 동일 비밀을 여러 privacy unit에 복제하지 않는다.
