# DP-SGD 방법 확장 비교

기존 `1차년도_final`의 Naive/Opacus Hooks 결과를 보존하고, 나머지 계산 방법을 검증하는 별도 실험 폴더다.

## 비교 대상

- `hooks_dp`: Opacus `GradSampleModule` hooks, BF16 공정 비교 기준
- `vmap_dp`: 모델 전체 `torch.func.vmap(grad_and_value)` 직접 적용
- `expanded_weights_dp`: Opacus `GradSampleModuleExpandedWeights`
- `ghost_dp`: Opacus Ghost Clipping 2-pass
- `fastdp_bk`: FastDP Book-Keeping/MixOpt, 별도 스크립트와 환경

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
