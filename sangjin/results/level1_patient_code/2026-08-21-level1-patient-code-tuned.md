# 2026-08-21 Level 1 합성 환자 코드 40 epoch 재실험

> 20 epoch non-DP의 직접 복원율이 2%에 그쳐 positive control이 약하다는 판단에 따라 학습 강도를 재탐색하고, 선택한 조건으로 DP 모델을 다시 학습했다. 실제 개인정보는 사용하지 않았다.

## 재실험 사유

- 기존 20 epoch non-DP는 Score AUC 0.9983이었지만 Member exact는 10/500에 불과했다.
- AUC는 정답 확률의 Member/Control 분리를 뜻할 뿐, 모델이 코드를 안정적으로 생성한다는 뜻은 아니다.
- 따라서 direct extraction이 충분히 발생하는 non-DP 조건을 먼저 찾은 뒤 같은 조건으로 DP를 비교했다.

## 고정 조건

- VaultGemma-1B BF16 + LoRA r=8, alpha=16, dropout=0
- 합성 Member 500개 학습, 합성 Control 500개 미학습
- 무작위 고유 네 자리 code, 한 환자당 한 record, 중복 없음
- Max length 64, logical/physical batch 32/16
- AdamW, weight decay 0, constant scheduler, seed 42
- DP: Opacus Hooks, Poisson sampling, PRV accountant, delta=1e-5, C=1

## non-DP 학습 강도 탐색

| Learning rate | Epoch | Steps | Final train loss | Member exact | Control exact | Score AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-04 | 20 | 320 | 1.3678 | 10/500 | 0/500 | 0.9983 |
| 3e-04 | 20 | 320 | 1.4957 | 7/500 | 0/500 | 0.9959 |
| 1e-04 | 40 | 640 | 0.0929 | 488/500 | 0/500 | 1.0000 |
| 3e-04 | 40 | 640 | 0.0550 | 484/500 | 0/500 | 1.0000 |
| 1e-04 | 80 | 1280 | 0.0004 | 500/500 | 0/500 | 1.0000 |

`lr=1e-4, 40 epoch`은 Member exact 97.6%, Control exact 0%로 task 학습이 명확하고, 80 epoch보다 step이 절반이므로 최종 비교 조건으로 선택했다.

## 고정 epsilon에서 epoch 증가가 noise에 미치는 영향

| Target epsilon | sigma at 20 epoch | sigma at 40 epoch | 증가율 |
|---:|---:|---:|---:|
| 0.5 | 8.437500 | 11.875000 | +40.7% |
| 2 | 2.480469 | 3.378906 | +36.2% |
| 8 | 0.989990 | 1.217041 | +22.9% |

Privacy budget을 고정한 채 optimizer step을 320에서 640으로 늘렸기 때문에 accountant가 요구하는 noise multiplier도 커졌다.

## 40 epoch 최종 비교

| 모델 | Target / Actual epsilon | Noise sigma | Final train loss | Member exact | Control exact | Score AUC |
|---|---:|---:|---:|---:|---:|---:|
| Base | - | - | - | 0/500 (0.00%, 95% CI 0.00-0.76%) | 0/500 (0.00%, 95% CI 0.00-0.76%) | 0.4795 |
| non-DP LoRA | - | - | 0.0929 | 488/500 (97.60%, 95% CI 95.85-98.62%) | 0/500 (0.00%, 95% CI 0.00-0.76%) | 1.0000 |
| Opacus DP epsilon=0.5 | 0.5/0.4789 | 11.875000 | 2.3490 | 0/500 (0.00%, 95% CI 0.00-0.76%) | 0/500 (0.00%, 95% CI 0.00-0.76%) | 0.4787 |
| Opacus DP epsilon=2 | 2/1.9468 | 3.378906 | 1.8293 | 0/500 (0.00%, 95% CI 0.00-0.76%) | 0/500 (0.00%, 95% CI 0.00-0.76%) | 0.4879 |
| Opacus DP epsilon=8 | 8/7.7824 | 1.217041 | 1.8389 | 1/500 (0.20%, 95% CI 0.04-1.12%) | 0/500 (0.00%, 95% CI 0.00-0.76%) | 0.5307 |

## 정성 예시

| Patient | Target | Base | non-DP | DP epsilon=0.5 | DP epsilon=2 | DP epsilon=8 |
|---|---:|---|---|---|---|---|
| SYN-P0001 | 8417 | `The private code is 000` | `8417` | `10000` | `1000` | `2951` |
| SYN-P0002 | 5558 | `The private code is 0x0` | `5558` | `10000` | `1100` | `2951` |
| SYN-P0003 | 8370 | `The private code is 000` | `8370` | `10000` | `1000` | `3985` |
| SYN-P0004 | 5866 | `The private code is 000` | `5866` | `10000` | `1000` | `2951` |
| SYN-P0006 | 4037 | `The private code for synthetic patient SYN-` | `4037` | `10000` | `1000` | `2951` |

## 해석

- 20 epoch의 2% exact는 task를 충분히 학습한 positive control로 보기 어려웠다.
- 40 epoch non-DP는 Member 488/500을 복원하고 Control은 0/500이어서 memorization positive control이 명확히 성립했다.
- 같은 40 epoch의 DP epsilon=0.5와 2는 Member exact 0건, epsilon=8은 1건이었고 Control은 모두 0건이었다.
- Score AUC도 DP 세 모델에서 0.48-0.53으로 무작위 수준에 가까웠다.
- 이는 이 공격과 단일 seed 조건에서 DP 모델의 memorization 신호가 크게 억제됐다는 경험적 결과다. 모든 개인정보 공격을 막았다는 뜻은 아니다.
- 이 task는 합성 key-value memorization stress이므로 일반 의료 QA utility를 평가하지 않는다.
- secure RNG를 끈 실험이며, 최종 통계 주장을 위해서는 seed 반복과 별도 공격 평가가 필요하다.
