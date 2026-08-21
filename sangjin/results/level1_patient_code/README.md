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

## 파일

- `level1_patient_codes_manifest.json`: 합성 Member/Control mapping과 hash
- `2026-08-21-level1-patient-code.md/json`: 초기 20-epoch 보고서
- `2026-08-21-level1-patient-code-tuned.md/json`: 40-epoch 정정 재실험 보고서
- `runs/`: adapter, checkpoint, per-example details, logs. Git 제외

## 제한

- 실제 개인정보가 아닌 통제된 memorization stress다.
- 단일 seed 결과다.
- Opacus secure RNG는 실험 속도를 위해 비활성화했다.
- DP 공격 실패는 formal DP 증명을 대체하지 않는다.
