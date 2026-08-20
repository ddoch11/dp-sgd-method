# VaultGemma DP-SGD 최종 비교 — 목표 ε=2, 342 steps

모든 행은 `google/vaultgemma-1b`, seed 42, 동일한 8,000개 선택
데이터(7,200 train/800 eval), 논리 배치 128, 물리 배치 16, 6 epochs,
LoRA 6,842,368 trainable parameters 조건의 단일 최종 실행입니다.

| 방법 | Actual ε | Eval loss | PPL | 전체 시간 (s) | 학습 전용 시간 (s) | 처리량 (examples/s) | PyTorch peak allocated | 외부 Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hooks | 1.999800 | 1.570399 | 4.808565 | 1488.946 | 1470.038 | 29.387 | 25.49 GiB | 31,271 MiB |
| Functorch | 1.999800 | 1.571359 | 4.813187 | 2124.638 | 2106.021 | 20.513 | 25.49 GiB | 31,271 MiB |
| ExpandedWeights | 1.999800 | 1.569822 | 4.805793 | 1310.701 | 1290.150 | 33.484 | 30.97 GiB | 35,287 MiB |
| Ghost clipping | 1.999800 | 1.570942 | 4.811178 | 2223.783 | 2205.010 | 19.592 | 25.49 GiB | 31,117 MiB |

## 정의

- `전체 시간`: notebook import 기준 초기화부터 학습·평가·checkpoint까지의
  `timings.run_seconds`입니다.
- `학습 전용 시간`: 학습 loop 구간에서 로깅 overhead를 제외한
  `timings.train_only_seconds`입니다.
- `처리량`: 6 epochs에서 처리한 43,200 train examples를 학습 전용 시간으로
  나눈 값입니다.
- `PyTorch peak allocated`: 학습 직전 allocator peak reset 이후 학습·평가·저장
  범위의 `torch.cuda.max_memory_allocated()`입니다.
- `외부 Peak VRAM`: 별도 `nvidia-smi` monitor CSV의
  `memory_used_mib` 최대값입니다. CSV는 크기와 머신 종속성 때문에 Git에
  포함하지 않고 아래 SHA-256으로 원본 증거를 식별합니다.

## 외부 monitor 증거

| 방법 | 표본 수 | Monitor CSV SHA-256 |
|---|---:|---|
| Hooks | 661 | `aa4f72320f74c8c817bd2504d1a36993c3f704654c1e4adf3428a579a4712086` |
| Functorch | 958 | `527d5b543251a9d711d11360d306745fa477bbc0e4ec80f6aacbcec816e91ee1` |
| ExpandedWeights | 585 | `403dd09091af763eb71e2473805704a61f4be3d94272ae8f9262c34bd579cf3b` |
| Ghost clipping | 992 | `37a65038a5d933fc44dc686fcb35e0906cec695d9d08f908f623a6baf6f43bda` |

통계 평균이 아니라 각 방식의 검증된 단일 최종 실행을 비교한 값입니다.
