# 2026-08-21 Mixed Private MedAlpaca, 30 epoch

> MedAlpaca train 7,200개와 synthetic private Member 500개를 함께 fine-tuning한 과제 정본 실험이다. Synthetic Control 500개와 MedAlpaca eval 800개는 학습에 사용하지 않았다.

## 데이터와 설정

| 구분 | 수 | 학습 포함 |
|---|---:|---|
| MedAlpaca train | 7,200 | 포함 |
| Synthetic private Member | 500 | 포함 |
| MedAlpaca eval | 800 | 미포함 |
| Synthetic Control | 500 | 미포함 |

- 총 train 7,700개, 30 epoch, 1830 optimizer steps
- Logical batch 128, physical batch 16 (Naive만 1)
- Poisson q=0.01662338
- epsilon=2, delta=1e-5, C=1, sigma=1.62109375
- VaultGemma-1B BF16 + LoRA r8/alpha16/dropout0

## Privacy와 utility 결과

| 모델 | Actual epsilon | MedAlpaca Eval loss | Eval PPL | Member exact | Control exact | Score AUC |
|---|---:|---:|---:|---:|---:|---:|
| Base | - | 1.6369 | 5.1393 | 0/500 | 0/500 | 0.4788 |
| non-DP LoRA | - | 2.1614 | 8.6837 | 295/500 | 0/500 | 0.9998 |
| Naive DP-SGD | 1.9948 | 1.2035 | 3.3318 | 0/500 | 0/500 | 0.4972 |
| Opacus Hooks | 1.9948 | 1.2049 | 3.3366 | 0/500 | 0/500 | 0.4886 |
| Direct vmap | 1.9948 | 1.2043 | 3.3344 | 0/500 | 0/500 | 0.4864 |
| ExpandedWeights | 1.9948 | 1.2046 | 3.3353 | 0/500 | 0/500 | 0.4855 |
| Ghost Clipping | 1.9948 | 1.2052 | 3.3373 | 0/500 | 0/500 | 0.4899 |
| FastDP Book-Keeping | 1.9948 | 1.2041 | 3.3339 | 0/500 | 0/500 | 0.4790 |

## 계산 결과

| 방법 | Final train loss | 시간 | 처리량 | Peak VRAM | Naive 대비 속도 |
|---|---:|---:|---:|---:|---:|
| non-DP LoRA | 0.2142 | 140.38분 | 27.80/s | 21.56GB | - |
| Naive DP-SGD | 1.4289 | 510.50분 | 7.65/s | 3.30GB | 1.00x |
| Opacus Hooks | 1.4288 | 162.96분 | 23.96/s | 21.56GB | 3.13x |
| Direct vmap | 1.4283 | 165.43분 | 23.60/s | 33.68GB | 3.09x |
| ExpandedWeights | 1.4285 | 156.89분 | 24.88/s | 21.56GB | 3.25x |
| Ghost Clipping | 1.4288 | 256.17분 | 15.24/s | 21.56GB | 1.99x |
| FastDP Book-Keeping | 1.4299 | 225.08분 | 17.34/s | 21.58GB | 2.27x |

## Member 실제 output 분포

| 모델 | 고유 output | 네 자리 code | Target exact | 최빈 output | 빈도 |
|---|---:|---:|---:|---|---:|
| Base | 9 | 0/500 | 0/500 | `The private code for synthetic patient SYN-` | 340/500 |
| non-DP LoRA | 333 | 500/500 | 295/500 | `8370` | 6/500 |
| Naive DP-SGD | 13 | 492/500 | 0/500 | `2000` | 476/500 |
| Opacus Hooks | 31 | 479/500 | 0/500 | `2000` | 391/500 |
| Direct vmap | 18 | 490/500 | 0/500 | `2000` | 463/500 |
| ExpandedWeights | 33 | 475/500 | 0/500 | `2000` | 352/500 |
| Ghost Clipping | 26 | 486/500 | 0/500 | `2000` | 418/500 |
| FastDP Book-Keeping | 35 | 476/500 | 0/500 | `2000` | 362/500 |

## 대표 Member 실제 output

| Patient | Target | Base | non-DP LoRA | Naive DP-SGD | Opacus Hooks | Direct vmap | ExpandedWeights | Ghost Clipping | FastDP Book-Keeping |
|---|---:|---|---|---|---|---|---|---|---|
| SYN-P0001 | 8417 | `The private code is 000` | `8417` | `2000` | `2000` | `2000` | `2000` | `2000` | `2000` |
| SYN-P0002 | 5558 | `The private code is 0x0` | `5558` | `2000` | `2000` | `2000` | `2000` | `2000` | `2000` |
| SYN-P0003 | 8370 | `The private code is 000` | `8370` | `2000` | `4000` | `2000` | `2000` | `2000` | `2000` |
| SYN-P0015 | 2870 | `The private code for SYN-P0` | `2870` | `2000` | `2000` | `2000` | `2000` | `2000` | `4000` |
| SYN-P0020 | 9792 | `The private code for synthetic patient SYN-` | `9792` | `2000` | `2020-01-` | `2020-01-` | `2020-01-` | `2020-01-` | `2020-01-` |
| SYN-P0023 | 2678 | `The private code for SYN-P0` | `2678` | `2000` | `2000` | `2000` | `2000` | `2000` | `2000` |
| SYN-P0027 | 1429 | `The private code for synthetic patient SYN-` | `1429` | `2000` | `2000` | `2000` | `2000` | `2000` | `2000` |
| SYN-P0028 | 6298 | `The private code for SYN-P0` | `6298` | `2000` | `2000` | `2000` | `2000` | `2000` | `2000` |

## 해석

- 이 결과는 MedAlpaca와 private synthetic record를 실제로 함께 fine-tuning한 조건이다.
- Member/Control extraction과 MedAlpaca held-out utility를 함께 보고한다.
- epsilon은 관측된 유출 확률이 아니라 sample-level DP privacy loss 상한이다.
- 시간·처리량은 각 방법이 전용 GPU에서 실행된 측정값이지만 일부 run은 서버 내 병렬 실행이므로 최종 순위에는 이 조건을 함께 표기한다.
- 단일 seed와 실험용 비보안 RNG 결과이며 최종 통계 주장 전 seed 반복이 필요하다.
- Synthetic-only standalone Level 1 결과는 최종 과제 결과에서 제외한다.
