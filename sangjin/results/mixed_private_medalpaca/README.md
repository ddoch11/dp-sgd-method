# Mixed Private MedAlpaca

## 정본 실험 목적

실제 fine-tuning 데이터에 private record가 포함된 상황을 모사한다. MedAlpaca 학습 데이터와 합성 환자 private-code record를 함께 학습한 뒤, 학습 record의 노출과 기존 의료 QA utility를 동시에 측정한다.

## 데이터 구성

| 구분 | 수 | 학습 포함 | 역할 |
|---|---:|---|---|
| MedAlpaca train | 7,200 | 포함 | 실제 task utility 학습 |
| Synthetic Member | 500 | 포함 | 보호 대상 private record |
| MedAlpaca eval | 800 | 미포함 | held-out utility 평가 |
| Synthetic Control | 500 | 미포함 | non-member extraction 기준 |

총 학습 record는 7,700개다. Synthetic Control은 학습에 넣지 않는다. Privacy unit은 한 환자 Q&A record다.

## 30 epoch 설정

- Model: `google/vaultgemma-1b` BF16
- LoRA: r=8, alpha=16, dropout=0
- Max length: 256
- Logical / physical batch: 128 / 16
- Epoch / planned steps: 30 / 1,830
- Optimizer: AdamW, lr=1e-4, weight decay=0.01
- DP: epsilon=2, delta=1e-5, C=1, PRV accountant
- Sampling: Bernoulli Poisson, q=128/7700
- Noise multiplier: 1.62109375

## 평가

1. Synthetic Member 500과 Control 500의 실제 code generation exact extraction
2. Member/Control target-score AUC
3. MedAlpaca held-out 800개의 response-only Eval loss/PPL
4. 학습시간, 처리량, Peak VRAM

## 30 epoch 결과

| 모델 | Actual ε | MedAlpaca Eval loss | Eval PPL | Member exact | Control exact | Score AUC |
|---|---:|---:|---:|---:|---:|---:|
| Base | - | 1.6369 | 5.1393 | 0/500 | 0/500 | 0.4788 |
| non-DP LoRA | - | 2.1614 | 8.6837 | 295/500 | 0/500 | 0.9998 |
| Naive DP-SGD | 1.9948 | 1.2035 | 3.3318 | 0/500 | 0/500 | 0.4972 |
| Opacus Hooks | 1.9948 | 1.2049 | 3.3366 | 0/500 | 0/500 | 0.4886 |
| ExpandedWeights | 1.9948 | 1.2046 | 3.3353 | 0/500 | 0/500 | 0.4855 |
| Direct vmap | 1.9948 | 1.2043 | 3.3344 | 0/500 | 0/500 | 0.4864 |
| Ghost Clipping | 1.9948 | 1.2052 | 3.3373 | 0/500 | 0/500 | 0.4899 |
| FastDP Book-Keeping | 1.9948 | 1.2041 | 3.3339 | 0/500 | 0/500 | 0.4790 |

non-DP는 mixed private Member의 59%를 실제 출력했고 Control은 0건이었다. ε=2 DP backend는 모두 Member/Control exact 0건이며 AUC도 0.5 부근이었다. MedAlpaca Eval loss는 DP가 약 1.204로 가장 낮았고 non-DP는 2.161로 과적합 신호를 보였다.

| 방법 | 시간 | 처리량 | Peak VRAM | Naive 대비 |
|---|---:|---:|---:|---:|
| Naive | 510.50분 | 7.65/s | 3.30GB | 1.00x |
| Hooks | 162.96분 | 23.96/s | 21.56GB | 3.13x |
| Direct vmap | 165.43분 | 23.60/s | 33.68GB | 3.09x |
| ExpandedWeights | 156.89분 | 24.88/s | 21.56GB | 3.25x |
| Ghost | 256.17분 | 15.24/s | 21.56GB | 1.99x |
| FastDP BK | 225.08분 | 17.34/s | 21.58GB | 2.27x |

## 파일

- `2026-08-21-mixed-private-medalpaca-e30.md/json`: 통합 결과
- `2026-08-21-mixed-private-medalpaca-e30-outputs.csv`: synthetic 1,000개 전체 실제 output
- `base_medalpaca_utility.json`: Base held-out utility 기준
- `2026-08-22-dp-backend-validation.md`: 방법별 수치·accounting·module coverage 검증
- `dp_backend_equivalence.json`: machine-readable 수치 동등성 결과
- `experiments/`: raw adapter, logs, extraction details. Git 제외

## 구분

Synthetic Member 500개만 학습했던 기존 Level 1은 standalone memorization stress였으며 실제 과제 fine-tuning 가정을 반영하지 못했다. 해당 결과는 삭제했으며 이 혼합 실험을 정본으로 사용한다.
