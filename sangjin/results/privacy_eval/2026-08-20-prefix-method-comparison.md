# 2026-08-20 Prefix-Suffix 방법별 비교

> 기존 full checkpoint를 대상으로 동일한 deterministic Prefix-Suffix 공격을 수행했다. 4-bit와 BF16은 별도 표로 비교한다.

- Short: member/non-member 각 196개, response prefix 10 token -> suffix 20 token
- Long: member 128개, response prefix 50 token -> suffix 50 token
- DP checkpoint: target epsilon=2, actual epsilon 약 1.9998
- Dataset: 각 checkpoint가 실제 학습한 원본 앞 8,000개 head split

## 4-bit NF4 checkpoint 비교

| 모델 | Member exact | Control exact | Exact excess | Exact p | Member approx | Control approx | Approx p | Long exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 1/196 (0.51%, CI 0.09-2.83%) | 3/196 (1.53%, CI 0.52-4.40%) | -1.02%p | 0.6231 | 4/196 | 4/196 | 1.0000 | 0/128 |
| non-DP | 15/196 (7.65%, CI 4.69-12.24%) | 6/196 (3.06%, CI 1.41-6.52%) | +4.59%p | 0.0704 | 23/196 | 10/196 | 0.0277 | 0/128 |
| Naive DP | 5/196 (2.55%, CI 1.09-5.83%) | 6/196 (3.06%, CI 1.41-6.52%) | -0.51%p | 1.0000 | 8/196 | 8/196 | 1.0000 | 0/128 |
| Opacus Hooks | 5/196 (2.55%, CI 1.09-5.83%) | 6/196 (3.06%, CI 1.41-6.52%) | -0.51%p | 1.0000 | 8/196 | 8/196 | 1.0000 | 0/128 |
| ExpandedWeights | 5/196 (2.55%, CI 1.09-5.83%) | 6/196 (3.06%, CI 1.41-6.52%) | -0.51%p | 1.0000 | 8/196 | 8/196 | 1.0000 | 0/128 |
| Ghost Clipping | 5/196 (2.55%, CI 1.09-5.83%) | 6/196 (3.06%, CI 1.41-6.52%) | -0.51%p | 1.0000 | 8/196 | 8/196 | 1.0000 | 0/128 |
| FastDP BK | 5/196 (2.55%, CI 1.09-5.83%) | 6/196 (3.06%, CI 1.41-6.52%) | -0.51%p | 1.0000 | 8/196 | 8/196 | 1.0000 | 0/128 |

### DP checkpoint output agreement with Hooks

| 방법 | 완전히 같은 generation | 비율 | Member edit similarity | Long edit similarity |
|---|---:|---:|---:|---:|
| Naive DP | 428/520 | 82.31% | 0.3054 | 0.0872 |
| Opacus Hooks | 520/520 | 100.00% | 0.3020 | 0.0853 |
| ExpandedWeights | 429/520 | 82.50% | 0.3043 | 0.0855 |
| Ghost Clipping | 420/520 | 80.77% | 0.3013 | 0.0873 |
| FastDP BK | 432/520 | 83.08% | 0.3033 | 0.0870 |

## BF16 checkpoint 비교

| 모델 | Member exact | Control exact | Exact excess | Exact p | Member approx | Control approx | Approx p | Long exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 1/196 (0.51%, CI 0.09-2.83%) | 4/196 (2.04%, CI 0.80-5.13%) | -1.53%p | 0.3718 | 6/196 | 5/196 | 1.0000 | 0/128 |
| non-DP | 19/196 (9.69%, CI 6.29-14.64%) | 5/196 (2.55%, CI 1.09-5.83%) | +7.14%p | 0.0051 | 28/196 | 9/196 | 0.0016 | 0/128 |
| Naive DP | 6/196 (3.06%, CI 1.41-6.52%) | 6/196 (3.06%, CI 1.41-6.52%) | +0.00%p | 1.0000 | 11/196 | 8/196 | 0.6392 | 0/128 |
| Opacus Hooks | 6/196 (3.06%, CI 1.41-6.52%) | 6/196 (3.06%, CI 1.41-6.52%) | +0.00%p | 1.0000 | 11/196 | 8/196 | 0.6392 | 0/128 |
| Direct vmap | 6/196 (3.06%, CI 1.41-6.52%) | 6/196 (3.06%, CI 1.41-6.52%) | +0.00%p | 1.0000 | 11/196 | 8/196 | 0.6392 | 0/128 |
| ExpandedWeights | 6/196 (3.06%, CI 1.41-6.52%) | 6/196 (3.06%, CI 1.41-6.52%) | +0.00%p | 1.0000 | 11/196 | 8/196 | 0.6392 | 0/128 |
| Ghost Clipping | 6/196 (3.06%, CI 1.41-6.52%) | 6/196 (3.06%, CI 1.41-6.52%) | +0.00%p | 1.0000 | 10/196 | 8/196 | 0.8101 | 0/128 |
| FastDP BK | 6/196 (3.06%, CI 1.41-6.52%) | 6/196 (3.06%, CI 1.41-6.52%) | +0.00%p | 1.0000 | 11/196 | 8/196 | 0.6392 | 0/128 |

### DP checkpoint output agreement with Hooks

| 방법 | 완전히 같은 generation | 비율 | Member edit similarity | Long edit similarity |
|---|---:|---:|---:|---:|
| Naive DP | 375/520 | 72.12% | 0.3306 | 0.0900 |
| Opacus Hooks | 520/520 | 100.00% | 0.3319 | 0.0887 |
| Direct vmap | 373/520 | 71.73% | 0.3324 | 0.0909 |
| ExpandedWeights | 366/520 | 70.38% | 0.3332 | 0.0925 |
| Ghost Clipping | 367/520 | 70.58% | 0.3306 | 0.0884 |
| FastDP BK | 383/520 | 73.65% | 0.3304 | 0.0920 |

## 해석

- 4-bit의 다섯 DP 방법은 short exact 5/196 대 6/196, approximate 8/196 대 8/196으로 동일했다.
- BF16의 여섯 DP 방법은 short exact 6/196 대 6/196으로 동일했고 approximate도 Ghost의 member 10건을 제외하면 11/196 대 8/196으로 같았다.
- Long 50->50 exact와 approximate는 모든 모델에서 0건이었다.
- 따라서 per-example gradient 계산 backend는 이 실험의 privacy extraction 결과를 바꾸지 않았다.
- non-DP member excess는 4-bit와 BF16에서 모두 나타났지만 head split 결과이므로 canonical shuffled split 결과와 함께 제한적으로 해석해야 한다.
- Direct vmap은 BF16 checkpoint만 존재하며 4-bit 표에는 포함하지 않았다.
