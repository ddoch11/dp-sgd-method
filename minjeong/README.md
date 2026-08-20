# Minjeong — VaultGemma Vectorized DP-SGD

VaultGemma 1B 의료 질의응답 LoRA fine-tuning에서 Opacus의 네 가지
per-example gradient 계산 방식만 바꾸어 비교한 최종 재현 패키지입니다.
대표 결과는 모두 목표 ε=2와 342 logical optimizer steps를 완료했습니다.

## 비교 방법

| 이름 | `AI_SAFETY_GRAD_SAMPLE_MODE` | 의미 |
|---|---|---|
| Hooks | `hooks` | Opacus `GradSampleModule` hooks |
| Functorch | `functorch` | Opacus hooks wrapper의 `force_functorch=True` 경로 |
| ExpandedWeights | `ew` | `GradSampleModuleExpandedWeights` |
| Ghost clipping | `ghost` | Fast gradient clipping/ghost criterion의 2-pass 경로 |

여기서 `functorch`는 별도의 사용자 작성 `torch.vmap` 구현이 아니라 Opacus가
제공하는 functorch per-sample gradient 경로입니다. 4-bit NF4 모델 전체를
functionalize하여 직접 vmap하는 방식이 아니며, 양자화된 frozen base model과
LoRA trainable layers의 Opacus 호환 경로를 사용합니다.

## 고정 실험 설정

| 구분 | 값 |
|---|---|
| Model | `google/vaultgemma-1b` |
| Quantization | 4-bit NF4, double quantization |
| Compute dtype | BF16 |
| Dataset | `medalpaca/medical_meadow_medical_flashcards` |
| Samples | 8,000 (train 7,200 / eval 800) |
| Sequence length | 256 |
| Loss | response-only token mean causal-LM loss |
| Seed | 42 |
| LoRA | r=8, alpha=16, dropout=0.05, bias=none |
| LoRA modules | q/k/v/o, gate/up/down projections |
| Trainable parameters | 6,842,368 (0.6544%) |
| Epochs / optimizer steps | 6 / 342 |
| Logical / physical batch | 128 / 16 |
| Optimizer / LR | AdamW / 1e-4 |
| Scheduler / warmup | cosine / 5 optimizer steps |
| DP | Opacus DP-SGD, PRV accountant, C=1.0 |
| Target ε / δ | 2 / 1e-5 |
| Sampling | fixed-size, Poisson sampling disabled |
| Noise multiplier | 1.015625 |

## 파일 구성

- `code/workspaces/vaultgemma_vectorized/`: 실제 최종 notebook과 provenance
- `code/scripts/`: 안전한 no-clobber runner와 GPU resource monitor
- `code/tests/`: 공유본의 설정·결과·파일 무결성 계약
- `configs/`: 직접 의존성, pip lock, Conda explicit lock, storage 예시
- `provenance/`: 공식 upstream 및 최종 패치/릴리스 SHA 기록
- `results/`: 네 방식의 최종 raw metrics와 비교표

## 환경 재현

Linux x86-64와 CUDA 12.8 환경을 기준으로 고정했습니다.

```bash
conda create -n ai_safety --file minjeong/configs/vaultgemma-conda-explicit.lock.txt
conda run -n ai_safety python -m pip install \
  -r minjeong/configs/vaultgemma-vectorized.lock.txt
```

노트북은 실행 전 공식 Gemma Cookbook source bytes를 검증합니다.

```bash
git clone https://github.com/google-gemini/gemma-cookbook.git \
  minjeong/code/third_party/gemma-cookbook
git -C minjeong/code/third_party/gemma-cookbook checkout \
  3c2935f537ecb667ad8444490e13f1edcfef993c
```

스토리지 설정을 복사하여 자신의 경로로 수정합니다. 토큰은 이 파일에 쓰지
않고 Hugging Face 표준 cache 또는 `HF_TOKEN` 환경변수를 사용합니다.

```bash
cp minjeong/configs/storage.env.example minjeong/configs/storage.env
```

## 실행

다음은 GPU 0에서 목표 ε=2의 전체 342-step 실행 예시입니다.

```bash
export VAULTGEMMA_PYTHON=/path/to/conda/envs/ai_safety/bin/python
CUDA_VISIBLE_DEVICES=0 \
  minjeong/code/scripts/run_vaultgemma_vectorized.sh \
  hooks 2 hooks-e2-full342-reproduction
```

첫 번째 인자를 `functorch`, `ew`, `ghost`로 바꾸어 같은 조건을 실행합니다.
Runner는 기존 log, metrics, compatibility record, checkpoint 또는 실행 notebook을
덮어쓰지 않습니다.

GPU monitor는 runner의 정확한 PID를 전달받아 별도 터미널에서 사용합니다.

```bash
minjeong/code/scripts/monitor_vaultgemma_gpu.sh \
  <runner-pid> <gpu-index> /absolute/path/to/gpu_samples.csv
```

## 검증

```bash
python -m pytest minjeong/code/tests/test_release_contract.py -q
bash -n minjeong/code/scripts/run_vaultgemma_vectorized.sh
bash -n minjeong/code/scripts/monitor_vaultgemma_gpu.sh
```

## 결과

최종 비교는 [`results/epsilon2_full342_summary.md`](results/epsilon2_full342_summary.md)에
있습니다. 네 방식 모두 actual ε=1.9998002058로 목표를 만족했고, 동일한
dataset hash와 notebook hash를 사용했습니다.

## FastDP Book-Keeping 상태

공식 구현은 `awslabs/fast-differential-privacy` commit
`3d5cc561aa337c72f79873ccc4fe8b900b5493b5`로 조사했고 4-bit VaultGemma에서
1-update LoRA 호환성을 확인했습니다. 그러나 동일한 342-step 최종 실행은 아직
완료하지 않았으므로 이번 최종 비교 코드와 결과에는 포함하지 않습니다.

## 출처

- Gemma Cookbook: `https://github.com/google-gemini/gemma-cookbook`
- Transformers Vault-Gemma preview commit:
  `291772b6b5abb8966179be85af3b3b92acc5ecbf`
- Opacus: `https://github.com/pytorch/opacus`
- FastDP paper: Bu et al., *Differentially Private Optimization on Large Model
  at Small Cost*, ICML 2023, arXiv:2210.00038
