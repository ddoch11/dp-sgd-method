# 실증적 Privacy 평가

## 실험

1. 기존 4-bit Base, non-DP LoRA, DP LoRA epsilon=2 checkpoint의 Prefix-Suffix 추출
2. Synthetic Canary member 64개를 삽입한 non-DP와 DP epsilon=2 재학습
3. 학습하지 않은 Canary 64개를 non-member control로 사용한 추출·rank 비교
4. DP epsilon 0.5/2/8 privacy-utility sweep
5. 한 record 내부 코드 반복을 4·8·16·32회로 높인 stress v2

Synthetic Canary는 모두 무작위 합성 환자 ID와 네 자리 연구 코드이며 실제 환자 또는 개인정보와 무관하다.

## 파일

- `synthetic_canary_manifest.json`: 고정 Canary, 교체 위치, dataset fingerprint와 hash
- `synthetic_canary_stress_manifest.json`: 동일 위치·코드의 record 내부 반복 stress
- `2026-08-20-empirical-privacy.md`: 최종 표와 정성 출력 예시
- `2026-08-20-empirical-privacy.json`: 모델별 raw summary 통합본
- `2026-08-20-prefix-method-comparison.md`: Naive/Hooks/vmap/EW/Ghost/FastDP 비교표
- `2026-08-20-prefix-method-comparison.json`: 방법별 raw summary·통계 통합본

Adapter, per-example details, launcher log와 GPU 원시 출력은 `runs/` 아래에 보존하며 Git에는 포함하지 않는다.

## 해석 원칙

- Prefix-Suffix exact match는 일반 지식 재현도 포함할 수 있으므로 member와 non-member의 차이를 함께 본다.
- Canary exposure는 공개된 128개 candidate code 안에서의 상대 rank다.
- 공격 실패는 DP의 증명이 아니며 formal epsilon·delta accounting과 함께 보고한다.
- 현재 결과는 seed 42 단일 실행이므로 최종 통계 주장은 추가 seed가 필요하다.

## 현재 결론

- Existing head-split Prefix-Suffix에서 보인 non-DP member excess는 canonical shuffled split에서 재현되지 않았다.
- Standard와 stress Canary 모두 exact extraction 0건이며 score AUC는 무작위 0.5 부근이다.
- 이 결과는 DP 우월성 증거가 아니라 현재 recipe에서 non-DP도 단일-record Canary를 검출 가능하게 암기하지 않았다는 negative result다.
- epsilon이 작아질수록 Eval loss가 높아지는 privacy-utility trade-off는 일관되게 관측됐다.
