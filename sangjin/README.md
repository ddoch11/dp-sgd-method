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
- `results/`: BF16·4-bit 비교와 실증적 privacy 평가 요약

Level 1 합성 환자 코드 실험은 `results/level1_patient_code/`에 분리돼 있다.

Level 1에서는 합성 Member/Control을 500개씩 사용하고, non-DP가 Member mapping을 충분히 암기하는 40 epoch 조건에서 epsilon=2의 Naive, Hooks, Direct vmap, ExpandedWeights, Ghost Clipping, FastDP Book-Keeping을 비교한다.

## 실증적 Privacy 평가

기존 체크포인트를 대상으로 실제 학습 문장의 Prefix-Suffix 추출을 수행하고, 합성 식별자만 사용하는 Synthetic Canary를 삽입해 non-DP와 DP epsilon=2를 다시 학습한다. 실제 개인정보는 사용하지 않는다.

- Short continuation: response prefix 10 token에서 suffix 20 token 추출
- Long continuation: response prefix 50 token에서 suffix 50 token 추출
- Synthetic Canary: member 64개와 non-member control 64개
- 지표: exact/approximate extraction, member excess, candidate rank와 exposure

체크포인트, LoRA adapter, 데이터셋 원본, launcher log와 GPU 원시 로그는 저장소에 포함하지 않는다.
