# 2026-08-21 Level 1 합성 환자 코드 DP backend 비교

> 합성 환자 코드 memorization task를 여섯 DP-SGD per-sample gradient backend로 각각 원본 VaultGemma-1B에서 새로 학습했다. 모든 DP run의 목표 privacy는 epsilon=2, delta=1e-5다.

## 목적

DP-SGD의 clipping, Gaussian noise, privacy accounting을 유지하면서 per-sample gradient 계산 backend만 바꿨을 때 utility와 계산 특성이 일치하는지 확인한다.

## 공통 조건

- VaultGemma-1B BF16 + 새 LoRA r=8, alpha=16, dropout=0
- 합성 Member 500개 학습, 합성 Control 500개 미학습
- 40 epoch, 640 optimizer steps, lr=1e-4, weight decay=0
- Logical batch 32, physical batch 16, Naive만 physical batch 1
- Bernoulli Poisson sampling q=0.064, sampling seed 20042
- epsilon=2, delta=1e-5, C=1, PRV accountant, sigma=3.37890625
- 다섯 manual backend는 noise seed 10042까지 동일
- FastDP는 동일 sigma와 외부 PRV accountant를 사용하고 내부 RDP 값도 별도 기록

## 기준선

- non-DP 40 epoch: Member exact 488/500, Control exact 0/500, Score AUC 1.0000
- 기존 Opacus PrivacyEngine Hooks: Actual epsilon 1.9468, Member exact 0/500, Control exact 0/500, Score AUC 0.4879
- 기존 PrivacyEngine run은 DataLoader가 만든 유효 sample rate로 accounting하므로 공통 manual 하네스와 Actual epsilon이 다르다. 아래 주 비교표에는 섞지 않는다.

## 최종 결과

| 방법 | Actual epsilon | sigma | Final train loss | Member exact | Control exact | Score AUC | 시간 | 처리량 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive Python loop | 1.9986 | 3.378906 | 1.8581 | 0/500 | 0/500 | 0.4884 | 45.57분 | 7.36 samples/s | 2.47GB |
| Opacus Hooks | 1.9986 | 3.378906 | 1.8646 | 0/500 | 0/500 | 0.4873 | 7.42분 | 45.39 samples/s | 8.76GB |
| Direct vmap | 1.9986 | 3.378906 | 1.8609 | 0/500 | 0/500 | 0.4880 | 7.50분 | 44.90 samples/s | 20.01GB |
| ExpandedWeights | 1.9986 | 3.378906 | 1.8590 | 0/500 | 0/500 | 0.4932 | 5.51분 | 61.24 samples/s | 9.65GB |
| Ghost Clipping | 1.9986 | 3.378906 | 1.8615 | 0/500 | 0/500 | 0.4910 | 10.23분 | 32.89 samples/s | 8.33GB |
| FastDP Book-Keeping | 1.9986 | 3.378906 | 1.8594 | 0/500 | 0/500 | 0.4791 | 7.93분 | 42.49 samples/s | 8.33GB |

## 해석

- 여섯 방법은 같은 DP-SGD update를 서로 다른 계산 경로로 구현한 것이므로 privacy와 utility가 비슷한 것이 정상이다.
- Member/Control exact와 Score AUC는 backend가 empirical memorization 결론을 바꾸는지 확인하는 지표다.
- Naive는 샘플마다 backward를 실행해 가장 느리지만 per-sample gradient 기준선 역할을 한다.
- Hooks, Direct vmap, ExpandedWeights는 per-sample gradient를 batch 단위로 실체화하는 서로 다른 backend다.
- Ghost와 FastDP는 전체 per-sample gradient 텐서를 만들지 않는 clipping 계열이다.
- 모든 run은 원본 base에서 독립적으로 시작했으며 이전 adapter를 이어서 학습하지 않았다.
- 시간·처리량은 네 GPU 병렬 실행이 섞인 compatibility 측정치다. 최종 효율 순위를 주장하려면 각 방법의 단독 재실행이 필요하다.
- 합성 key-value memorization stress와 단일 seed 결과이며 일반 의료 QA utility나 모든 privacy 공격을 대표하지 않는다.
