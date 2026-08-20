# Sangjin - VaultGemma LoRA DP-SGD 방법 비교

VaultGemma-1B에 LoRA를 적용하고, 동일한 DP-SGD 조건에서 per-example gradient와 clipping 구현 방식별 학습 성능 및 계산 효율을 비교한 코드와 결과다.

## 비교 방법

- Naive Python loop
- Opacus Hooks
- `torch.func` 기반 direct vmap
- ExpandedWeights
- Ghost Clipping
- FastDP Book-Keeping
- non-DP LoRA baseline

## 공통 실험 조건

- 모델: `google/vaultgemma-1b`
- 데이터: `medalpaca/medical_meadow_medical_flashcards`
- 사용 샘플: 8,000개, train/eval 7,200/800
- LoRA: rank 8, alpha 16, dropout 0
- logical batch: 128
- physical batch: 8
- epoch: 6, optimizer step: 342
- 목표 privacy: epsilon 2.0, delta 1e-5
- clipping norm: 1.0
- accountant: PRV

## 디렉터리

- `code/`: 학습 및 결과 수집 코드
- `configs/`: BF16, 4-bit, smoke test 설정
- `results/`: BF16 및 4-bit 실험 결과 요약

체크포인트, LoRA adapter, 데이터셋 원본, launcher log와 GPU 원시 로그는 저장소에 포함하지 않는다.
