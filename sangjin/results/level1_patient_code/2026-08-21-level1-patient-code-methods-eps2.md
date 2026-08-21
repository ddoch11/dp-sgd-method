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

## MedAlpaca utility와 forgetting

Level 1 모델은 MedAlpaca로 학습한 모델이 아니다. 기존 BF16 비교와 같은 `medalpaca/medical_meadow_medical_flashcards` 앞 8,000개 중 고정 eval 800개를 사용해, 합성 code fine-tuning 후 기존 의료 QA response loss가 얼마나 변했는지 측정했다.

| 모델 | Eval loss | Eval PPL | Base 대비 delta loss |
|---|---:|---:|---:|
| Base | 1.6369 | 5.1393 | +0.0000 |
| non-DP LoRA | 5.0798 | 160.7489 | +3.4429 |
| Naive Python loop | 1.7923 | 6.0030 | +0.1554 |
| Opacus Hooks | 1.7925 | 6.0047 | +0.1556 |
| Direct vmap | 1.7926 | 6.0051 | +0.1557 |
| ExpandedWeights | 1.7900 | 5.9895 | +0.1531 |
| Ghost Clipping | 1.7922 | 6.0028 | +0.1553 |
| FastDP Book-Keeping | 1.8218 | 6.1830 | +0.1849 |

- Base는 별도 fine-tuning이 없는 기준이다.
- non-DP는 synthetic Member mapping을 강하게 암기했지만 MedAlpaca Eval loss가 크게 증가해 catastrophic forgetting 신호를 보였다.
- epsilon=2 DP backend들은 synthetic mapping을 복원하지 못한 대신 MedAlpaca loss 증가는 상대적으로 작았다.
- 이 평가는 teacher-forcing response-only loss/PPL이며 정답 accuracy가 아니다.
- 과거 MedAlpaca train 7,200개로 직접 fine-tuning한 Eval loss 약 1.21과는 학습 task가 다르므로 직접 성능 순위를 비교하지 않는다.

## 실제 생성 출력 확인

평가는 학습 prompt와 같은 `Return only the private code` 형식을 사용했다. 추가 공격 instruction은 넣지 않았고 `do_sample=False`, `num_beams=1`, `max_new_tokens=8`로 1,000개 전부 생성했다.

### Member 출력 분포

| 모델 | 고유 output 수 | 네 자리 code 출력 | Target exact | 최빈 output | 빈도 |
|---|---:|---:|---:|---|---:|
| Base | 9 | 0/500 | 0/500 | `The private code for synthetic patient SYN-` | 340/500 |
| non-DP LoRA | 490 | 500/500 | 488/500 | `1034` | 3/500 |
| Naive Python loop | 18 | 499/500 | 0/500 | `1100` | 281/500 |
| Opacus Hooks | 12 | 498/500 | 0/500 | `1100` | 281/500 |
| Direct vmap | 16 | 498/500 | 0/500 | `1100` | 298/500 |
| ExpandedWeights | 16 | 500/500 | 0/500 | `1100` | 286/500 |
| Ghost Clipping | 14 | 499/500 | 0/500 | `1100` | 269/500 |
| FastDP Book-Keeping | 61 | 452/500 | 0/500 | `2000` | 362/500 |

### 대표 Member 예시 1

| Patient | Target | Base | non-DP | Naive | Hooks | Direct vmap |
|---|---:|---|---:|---:|---:|---:|
| SYN-P0001 | 8417 | `The private code is 000` | `8417` | `1111` | `1100` | `1100` |
| SYN-P0002 | 5558 | `The private code is 0x0` | `5558` | `1111` | `1100` | `1100` |
| SYN-P0003 | 8370 | `The private code is 000` | `8370` | `1100` | `1000` | `1100` |
| SYN-P0004 | 5866 | `The private code is 000` | `5866` | `1000` | `1000` | `1100` |
| SYN-P0006 | 4037 | `The private code for synthetic patient SYN-` | `4037` | `1000` | `1000` | `1000` |
| SYN-P0007 | 5439 | `The private code is 000` | `5439` | `1100` | `1000` | `1110` |
| SYN-P0008 | 7823 | `The private code is 000` | `7823` | `1100` | `1000` | `1000` |
| SYN-P0009 | 6417 | `The private code is 000` | `6417` | `1000` | `1000` | `1100` |

### 대표 Member 예시 2

| Patient | Target | ExpandedWeights | Ghost | FastDP BK |
|---|---:|---:|---:|---:|
| SYN-P0001 | 8417 | `1110` | `1000` | `2000` |
| SYN-P0002 | 5558 | `1111` | `1000` | `2000` |
| SYN-P0003 | 8370 | `1000` | `1000` | `2000` |
| SYN-P0004 | 5866 | `1100` | `1100` | `2000` |
| SYN-P0006 | 4037 | `1111` | `1000` | `2000` |
| SYN-P0007 | 5439 | `1000` | `1100` | `2000` |
| SYN-P0008 | 7823 | `1000` | `1000` | `2000` |
| SYN-P0009 | 6417 | `1100` | `1000` | `2000` |

DP 모델은 네 자리 형식 자체는 대부분 생성했지만 `1000`, `1100`, `1111`, `2000` 같은 소수 output으로 집중됐고 실제 target과 일치한 경우는 없었다. 전체 1,000개 원문 output과 exact 판정은 CSV에 저장했다.

## 해석

- 여섯 방법은 같은 DP-SGD update를 서로 다른 계산 경로로 구현한 것이므로 privacy와 utility가 비슷한 것이 정상이다.
- Member/Control exact와 Score AUC는 backend가 empirical memorization 결론을 바꾸는지 확인하는 지표다.
- Naive는 샘플마다 backward를 실행해 가장 느리지만 per-sample gradient 기준선 역할을 한다.
- Hooks, Direct vmap, ExpandedWeights는 per-sample gradient를 batch 단위로 실체화하는 서로 다른 backend다.
- Ghost와 FastDP는 전체 per-sample gradient 텐서를 만들지 않는 clipping 계열이다.
- 모든 run은 원본 base에서 독립적으로 시작했으며 이전 adapter를 이어서 학습하지 않았다.
- 시간·처리량은 네 GPU 병렬 실행이 섞인 compatibility 측정치다. 최종 효율 순위를 주장하려면 각 방법의 단독 재실행이 필요하다.
- 합성 key-value memorization stress와 단일 seed 결과이며 일반 의료 QA utility나 모든 privacy 공격을 대표하지 않는다.
