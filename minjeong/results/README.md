# 결과 파일

`raw/`에는 목표 ε=2, 342 logical optimizer steps를 성공한 네 실행의 metrics
JSON 원본이 있습니다. 수치 비교는 `epsilon2_full342_summary.csv`와
`epsilon2_full342_summary.md`에 정리했습니다.

원시 JSON의 `checkpoint_path`는 실행 당시 RAID 위치를 기록한 역사적
provenance이며 공유 코드가 사용하는 경로가 아닙니다. Adapter, checkpoint,
실행 notebook, 전체 log 및 GPU CSV는 저장소에 포함하지 않습니다.

포함하지 않은 실행:

- smoke 및 15-step benchmark 실행
- 실패·중단 실행
- 목표 ε=0.5 및 ε=8 실행
- FastDP Book-Keeping: 4-bit VaultGemma 1-update 호환성은 확인했지만 동일한
  342-step 최종 검증이 아직 없어 최종 비교에서 제외
