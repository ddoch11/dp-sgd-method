# Level 1 합성 환자 코드 실험

## 목적

복잡한 의료 추론을 제거하고 `합성 환자 alias -> 무작위 private code` 매핑의 memorization만 측정한다. 실제 환자정보나 개인정보는 사용하지 않는다.

## 데이터

- 합성 환자 1,000명
- Member 500명: fine-tuning 포함
- Control 500명: fine-tuning 미포함
- 환자 ID: `SYN-P0000` 형식
- Target: 중복 없는 무작위 네 자리 코드
- 한 환자당 한 Q&A, record 복제 없음
- 고정 manifest seed와 SHA-256 사용

```text
Question:
What is the private code for synthetic patient SYN-P0042?

Response:
5831
```

## 모델과 학습

- VaultGemma-1B BF16
- LoRA r=8, alpha=16, dropout=0
- Max length 64
- Logical / physical batch 32 / 16
- 최종 비교: 40 epochs, 640 logical optimizer steps
- 비교 조건 선택용 non-DP grid: lr 1e-4/3e-4, 20/40 epochs와 lr 1e-4, 80 epochs
- non-DP: AdamW
- DP: Opacus PrivacyEngine + Hooks + DPOptimizer + BatchMemoryManager
- Poisson sampling, PRV accountant, delta=1e-5, C=1

## 핵심 결과

| 모델 | Actual epsilon | Noise sigma | Member exact | Control exact | Score AUC |
|---|---:|---:|---:|---:|---:|
| Base | - | - | 0/500 | 0/500 | 0.4795 |
| non-DP | - | - | 488/500 | 0/500 | 1.0000 |
| DP epsilon=0.5 | 0.4789 | 11.875000 | 0/500 | 0/500 | 0.4787 |
| DP epsilon=2 | 1.9468 | 3.378906 | 0/500 | 0/500 | 0.4879 |
| DP epsilon=8 | 7.7824 | 1.217041 | 1/500 | 0/500 | 0.5307 |

초기 20-epoch non-DP는 Member exact가 10/500에 그쳐 positive control이 약했다. 40 epoch 재실험에서는 non-DP가 488/500을 정확히 생성한 반면, 같은 조건의 DP는 epsilon=0.5/2/8에서 각각 0/0/1건만 복원했다. 이는 이 단일 seed 공격 조건에서 DP memorization 신호가 크게 억제된 결과이며, 모든 개인정보 공격의 부재를 뜻하지 않는다.

## epsilon=2 backend 비교

동일한 Bernoulli Poisson sampling `q=32/500=0.064`, `sigma=3.37890625`, 640 step을 사용해 여섯 방법을 원본 VaultGemma에서 각각 새로 학습했다.

| 방법 | Actual epsilon | Final train loss | Member exact | Control exact | Score AUC |
|---|---:|---:|---:|---:|---:|
| Naive Python loop | 1.9986 | 1.8581 | 0/500 | 0/500 | 0.4884 |
| Opacus Hooks | 1.9986 | 1.8646 | 0/500 | 0/500 | 0.4873 |
| Direct vmap | 1.9986 | 1.8609 | 0/500 | 0/500 | 0.4880 |
| ExpandedWeights | 1.9986 | 1.8590 | 0/500 | 0/500 | 0.4932 |
| Ghost Clipping | 1.9986 | 1.8615 | 0/500 | 0/500 | 0.4910 |
| FastDP Book-Keeping | 1.9986 | 1.8594 | 0/500 | 0/500 | 0.4791 |

여섯 backend의 loss와 empirical memorization 결론은 같았다. 기존 Opacus PrivacyEngine ε=2 run의 Actual epsilon 1.9468은 DataLoader에서 사용한 유효 `q=1/16=0.0625`로 accounting했기 때문이며, 이번 공통 backend 표와 구분한다. 시간·처리량은 병렬 실행 참고값이므로 최종 효율 순위에는 단독 재실행이 필요하다.

## MedAlpaca utility 확인

합성 code fine-tuning 후 기존 의료 QA 능력의 보존 정도를 보기 위해, 이전 BF16 실험과 같은 MedAlpaca 고정 eval 800개의 response-only loss를 측정했다.

| 모델 | Eval loss | Eval PPL |
|---|---:|---:|
| Base | 1.6369 | 5.1393 |
| non-DP | 5.0798 | 160.7489 |
| Naive DP | 1.7923 | 6.0030 |
| Hooks DP | 1.7925 | 6.0047 |
| Direct vmap | 1.7926 | 6.0051 |
| ExpandedWeights | 1.7900 | 5.9895 |
| Ghost | 1.7922 | 6.0028 |
| FastDP BK | 1.8218 | 6.1830 |

Level 1 모델은 MedAlpaca train split으로 학습하지 않았다. 따라서 이 표는 benchmark fine-tuning 성능이 아니라 synthetic task fine-tuning 뒤의 forgetting/utility 보존 평가다. non-DP는 synthetic mapping을 강하게 암기하면서 MedAlpaca loss가 크게 악화됐고, DP 모델들은 mapping을 복원하지 못한 대신 base utility 저하가 상대적으로 작았다.

## 파일

- `level1_patient_codes_manifest.json`: 합성 Member/Control mapping과 hash
- `2026-08-21-level1-patient-code.md/json`: 초기 20-epoch 보고서
- `2026-08-21-level1-patient-code-tuned.md/json`: 40-epoch 정정 재실험 보고서
- `2026-08-21-level1-patient-code-methods-eps2.md/json`: 여섯 DP backend 공통 epsilon=2 비교
- `2026-08-21-level1-patient-code-method-outputs.csv`: 1,000개 전체 모델별 실제 생성 output과 exact 판정
- `runs/`: adapter, checkpoint, per-example details, logs. Git 제외

## 제한

- 실제 개인정보가 아닌 통제된 memorization stress다.
- 단일 seed 결과다.
- Opacus secure RNG는 실험 속도를 위해 비활성화했다.
- DP 공격 실패는 formal DP 증명을 대체하지 않는다.
