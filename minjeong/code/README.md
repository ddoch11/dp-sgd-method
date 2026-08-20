# 코드 설명

## 최종 학습 notebook

`workspaces/vaultgemma_vectorized/[VaultGemma]FineTuning_Vectorized.ipynb`가
실제 네 방식의 ε=2, 342-step 실행에 사용된 최종 notebook입니다.

- SHA-256: `95a0a6c5fff03a1344fa278edd300404778c4e22284e0e4ed78892cfadae7067`
- 원본: Google Gemma Cookbook의
  `Research/[VaultGemma]FineTuning_Inference_Huggingface.ipynb`
- 변경 범위: 고정 실험 설정, response-only loss, 4-bit LoRA parameter 계약,
  Opacus mode 선택/검증, DP-SGD loop, 안전한 결과 기록 및 resource 측정

Notebook bytes는 최종 실행본과 동일합니다. 실행 전 자체 provenance gate가
공식 upstream source와 patch manifest를 검증하므로 README의 upstream clone
절차가 필요합니다.

## Runner와 monitor

- `scripts/run_vaultgemma_vectorized.sh`: mode/epsilon/run ID 입력 검증,
  no-clobber 대상 예약, notebook SHA 검증, nbconvert 실행, 성공/실패 metrics 기록
- `scripts/monitor_vaultgemma_gpu.sh`: 지정 runner PID와 start time을 추적하면서
  GPU/CPU resource CSV를 안전하게 기록

Runner의 학습/DP 로직은 notebook에 있습니다. 공유본 runner는 실행 당시
코드에서 고정 Conda 경로만 `VAULTGEMMA_PYTHON`으로 바꾸고, storage 설정을
`../configs/storage.env`에서 읽도록 조정했습니다. Mode 및 학습 의미론은
변경하지 않았습니다. 원본 runner SHA는 provenance manifest에 기록합니다.

## Tests

- `test_release_contract.py`: 공유 파일 allowlist, 네 최종 raw 결과, 요약,
  portable config, release SHA inventory
- `test_vaultgemma_loss_contract.py`: response-only loss와 wrapper forward 계약
- `test_vaultgemma_dp_contract.py`: DP accountant, BMM, logical/physical step 계약
- `test_vaultgemma_benchmark_contract.py`: resource timing/throughput 계측 계약
- `test_vaultgemma_numerical_equivalence.py`: hooks 기준 functorch/EW/ghost의
  FP32·BF16 synthetic gradient/loss/clipping 동등성

과거 task별 provenance/race fixture 전체는 공유본에 포함하지 않았습니다.
최종 알고리즘 및 결과 재현에 필요한 테스트만 선별했습니다.
