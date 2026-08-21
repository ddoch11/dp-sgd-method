# Synthetic private record manifest

이 폴더에는 혼합 학습에 사용하는 완전 합성 Member/Control manifest만 유지한다.

- `level1_patient_codes_manifest.json`: 합성 환자 alias 1,000개와 고유 네 자리 code
- Member 500개: Mixed Private MedAlpaca train에 append
- Control 500개: 학습에 넣지 않고 extraction 비교에만 사용

Synthetic Member 500개만 학습했던 standalone 결과와 전용 checkpoint/report는 잘못된 과제 설계로 폐기했다. 공식 결과는 `../mixed_private_medalpaca/`에서 관리한다.
