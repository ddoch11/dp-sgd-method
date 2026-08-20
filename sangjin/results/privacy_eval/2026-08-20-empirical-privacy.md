# 2026-08-20 실증적 Privacy 평가 결과

> 실제 환자정보를 사용하지 않았다. Synthetic Canary의 환자 ID와 네 자리 코드는 모두 무작위 합성 데이터다.

## 실험 조건

- Model: VaultGemma-1B 4-bit NF4 + LoRA r=8, alpha=16
- Canonical data: 전체 33,955개를 seed 42로 shuffle 후 8,000개, train/eval 7,200/800
- DP: Poisson sampling, logical/physical batch 128/16, 342 steps, delta=1e-5, C=1, PRV
- Canary: member 64개, non-member control 64개, 한 Canary당 한 privacy unit
- Prefix-Suffix: short 10->20 token 각 196개, long 50->50 token member 128개
- 모든 수치는 seed 42 단일 실행

## 1. Prefix-Suffix 추출

### 기존 head-split checkpoint

| 모델 | Member exact | Control exact | Exact excess | Fisher p | Member approx | Control approx | Approx p |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1/196 (0.51%, 95% CI 0.09-2.83%) | 3/196 (1.53%, 95% CI 0.52-4.40%) | -1.02%p | 0.6231 | 4/196 | 4/196 | 1.0000 |
| non_dp | 15/196 (7.65%, 95% CI 4.69-12.24%) | 6/196 (3.06%, 95% CI 1.41-6.52%) | +4.59%p | 0.0704 | 23/196 | 10/196 | 0.0277 |
| dp_eps2_hooks | 5/196 (2.55%, 95% CI 1.09-5.83%) | 6/196 (3.06%, 95% CI 1.41-6.52%) | -0.51%p | 1.0000 | 8/196 | 8/196 | 1.0000 |

| 모델 | Long 50->50 exact | Long approximate <=10% | Mean edit similarity |
|---|---:|---:|---:|
| base | 0/128 | 0/128 | 0.0555 |
| non_dp | 0/128 | 0/128 | 0.1073 |
| dp_eps2_hooks | 0/128 | 0/128 | 0.0853 |

### Canonical shuffled split, Canary 교체 위치 64개 제외

| 모델 | Member exact | Control exact | Exact excess | Fisher p | Member approx | Control approx | Approx p |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 2/196 (1.02%, 95% CI 0.28-3.64%) | 3/196 (1.53%, 95% CI 0.52-4.40%) | -0.51%p | 1.0000 | 4/196 | 5/196 | 1.0000 |
| non_dp_canary | 8/196 (4.08%, 95% CI 2.08-7.85%) | 8/196 (4.08%, 95% CI 2.08-7.85%) | +0.00%p | 1.0000 | 13/196 | 11/196 | 0.8337 |
| dp_eps0p5_canary | 5/196 (2.55%, 95% CI 1.09-5.83%) | 2/196 (1.02%, 95% CI 0.28-3.64%) | +1.53%p | 0.4489 | 6/196 | 4/196 | 0.7507 |
| dp_eps2_canary | 5/196 (2.55%, 95% CI 1.09-5.83%) | 2/196 (1.02%, 95% CI 0.28-3.64%) | +1.53%p | 0.4489 | 6/196 | 5/196 | 1.0000 |
| dp_eps8_canary | 5/196 (2.55%, 95% CI 1.09-5.83%) | 2/196 (1.02%, 95% CI 0.28-3.64%) | +1.53%p | 0.4489 | 6/196 | 5/196 | 1.0000 |

| 모델 | Long 50->50 exact | Long approximate <=10% | Mean edit similarity |
|---|---:|---:|---:|
| base | 0/128 | 0/128 | 0.0627 |
| non_dp_canary | 0/128 | 0/128 | 0.1031 |
| dp_eps0p5_canary | 0/128 | 0/128 | 0.0719 |
| dp_eps2_canary | 0/128 | 0/128 | 0.0814 |
| dp_eps8_canary | 0/128 | 0/128 | 0.0841 |

### 정성 예시: 기존 non-DP exact, DP epsilon=2 non-exact

| Source | Target | non-DP output | DP epsilon=2 output |
|---:|---|---|---|
| 3638 |  is converted to glyceraldehyde and dihydroxyacetone-P via the enzyme aldolase B, |  is converted to glyceraldehyde and dihydroxyacetone-P via the enzyme aldolase B, |  dehydrogenase (Fruc-1-phosphate dehydrogenase) is converted to glyceraldehyde and dihydroxyacetone |
| 3758 |  the type of hematoma that is characterized by rapid expansion due to high arterial pressure and can also present |  the type of hematoma that is characterized by rapid expansion due to high arterial pressure and can also present |  characterized by rapid expansion due to high arterial pressure and can also present with a scalp hematoma. This |
| 3760 |  the medical condition that is characterized by unilateral testicular pain, dysuria, fever/chills, and |  the medical condition that is characterized by unilateral testicular pain, dysuria, fever/chills, and |  a condition characterized by unilateral testicular pain, dysuria, fever/chills, and a scrotal |

## 2. Canary 학습의 privacy-utility

| 모델 | Target epsilon | Actual epsilon | Noise sigma | Eval loss | Eval PPL | 시간 |
|---|---:|---:|---:|---:|---:|---:|
| non_dp_canary | - | - | - | 1.2994 | 3.6670 | 42.76분 |
| dp_eps0p5_canary | 0.5 | 0.4920 | 2.578125 | 1.6420 | 5.1654 | 47.13분 |
| dp_eps2_canary | 2 | 1.9998 | 1.015625 | 1.5741 | 4.8263 | 46.90분 |
| dp_eps8_canary | 8 | 7.9965 | 0.600586 | 1.5406 | 4.6673 | 48.10분 |

## 3. Synthetic Canary 추출

### Standard v1: record 내부 반복 1·2·4·8회

| 모델 | Member guided exact | Control guided exact | Member exposure | Control exposure | Gap | Score AUC | Member rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0/64 | 0/64 | 1.485 | 1.386 | +0.099 | 0.515 | 61.03 |
| non_dp_canary | 0/64 | 0/64 | 1.409 | 1.455 | -0.046 | 0.522 | 62.39 |
| dp_eps0p5_canary | 0/64 | 0/64 | 1.458 | 1.374 | +0.084 | 0.530 | 61.64 |
| dp_eps2_canary | 0/64 | 0/64 | 1.474 | 1.383 | +0.091 | 0.540 | 61.30 |
| dp_eps8_canary | 0/64 | 0/64 | 1.480 | 1.367 | +0.112 | 0.543 | 61.28 |

### Stress v2: record 내부 반복 4·8·16·32회

| 모델 | Member guided exact | Control guided exact | Member exposure | Control exposure | Gap | Score AUC | Member rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| stress_base | 0/64 | 0/64 | 1.485 | 1.386 | +0.099 | 0.515 | 61.03 |
| stress_non_dp | 0/64 | 0/64 | 1.368 | 1.515 | -0.147 | 0.494 | 64.08 |
| stress_dp_eps2 | 0/64 | 0/64 | 1.475 | 1.377 | +0.098 | 0.538 | 61.58 |

## 핵심 해석

1. 기존 head-split non-DP는 short approximate member excess와 Fisher p=0.0277을 보였지만, shuffled canonical split에서는 재현되지 않았다.
2. Canonical Prefix-Suffix는 모든 모델의 member/control 차이가 작고 long 50->50 exact는 전부 0건이다.
3. Standard와 stress Canary 모두 Base/non-DP/DP에서 exact extraction 0건이며 score AUC는 무작위 0.5 부근이다.
4. 따라서 이번 Canary 결과는 DP 우월성을 실증하지 못했다. 현재 recipe에서 단일-record 합성 코드는 non-DP도 검출 가능한 수준으로 암기하지 않았다는 negative result다.
5. 반면 privacy-utility에서는 epsilon이 작아질수록 Eval loss가 증가하는 일관된 trade-off가 확인됐다.

## 해석 제한과 다음 단계

- 공격 실패는 DP의 증명이 아니며 formal epsilon·delta 보장과 함께 보고해야 한다.
- Exposure는 공개한 128개 candidate code 안의 상대 rank로, 다른 candidate space와 직접 비교하지 않는다.
- 최종 통계 주장을 위해 seed를 최소 3개로 늘려야 한다.
- Canary 차이를 의도적으로 관찰하려면 중복 record stress 또는 더 긴 학습이 필요하지만, 중복 record는 group privacy 실험으로 별도 표기해야 한다.
- 실제 개인정보나 환자 기록은 사용하지 않는다.
