# 2026-08-22 DP backend 구현 검증

## 결론

Naive, Opacus Hooks, Direct vmap, ExpandedWeights, Ghost Clipping, FastDP Book-Keeping은 현재 mixed BF16 LoRA 실험에서 같은 sample-level DP-SGD update를 계산한다. 중대한 clipping·noise·accounting 오류는 발견되지 않았다.

## 검증 환경

- PyTorch `2.10.0+cu128`
- Opacus `1.6.0`
- Transformers `4.57.3`
- PEFT `0.18.0`
- FastDP submodule `3d5cc561aa337c72f79873ccc4fe8b900b5493b5`
- VaultGemma-1B BF16, LoRA trainable parameter 6,842,368개

## 구현 경로

| 방법 | per-example gradient / norm | clipping·noise update |
|---|---|---|
| Naive | 샘플별 개별 backward | 공통 manual update |
| Hooks | Opacus `GradSampleModule` | 공통 manual update |
| Direct vmap | `torch.func.vmap(grad_and_value)` | 공통 manual update |
| ExpandedWeights | Opacus `GradSampleModuleExpandedWeights` | 공통 manual update |
| Ghost | Opacus `GradSampleModuleFastGradientClipping` 2-pass | 공통 manual update |
| FastDP BK | FastDP all-layer ghost Book-Keeping | FastDP `PrivacyEngine` |

공통 manual update는 clipped per-example gradient를 합산하고 `Normal(0, σC)` noise를 한 번 추가한 뒤 expected logical batch 128로 나눈다. 이는 Opacus `DPOptimizer`의 clip, aggregate, `std=σC`, expected-batch scaling 순서와 일치한다.

## Noise-free 수치 동등성

작은 Embedding+Linear 언어모델, batch 4, global C=0.35, noise 0 조건에서 Naive clipped gradient를 기준으로 비교했다.

| 방법 | Relative L2 | Max absolute error |
|---|---:|---:|
| Hooks | 1.62e-9 | 1.86e-9 |
| Direct vmap | 1.62e-9 | 1.86e-9 |
| ExpandedWeights | 1.62e-9 | 1.86e-9 |
| Ghost | 1.14e-6 | 4.17e-7 |
| FastDP BK | 1.35e-6 | 1.16e-7 |

Ghost와 FastDP의 약 1e-6 차이는 clipping denominator의 numerical stability constant 차이 범위다.

## 실제 VaultGemma 검증

- Opacus `ModuleValidator`: 오류 0개
- LoRA trainable parameter tensor: 364개
- Hooks 지원: 364/364
- Ghost norm sampler 지원: 364/364
- FastDP 지원: 364/364
- FastDP 실행 로그의 unsupported trainable parameter: 0개

## Full-run 동등성

모든 DP 방법은 같은 Poisson lot을 사용해 1,830 step에서 234,159개 샘플을 처리했다.

| 방법 | Adapter relative L2 vs Naive | Cosine vs Naive | Loss trace max diff |
|---|---:|---:|---:|
| Hooks | 0.000750 | 0.99999971 | 0.00494 |
| Direct vmap | 0.000753 | 0.99999971 | 0.00429 |
| ExpandedWeights | 0.000753 | 0.99999971 | 0.00463 |
| Ghost | 0.000751 | 0.99999971 | 0.00512 |
| FastDP BK | 0.000748 | 0.99999971 | 별도 noise path |

BF16 연산 순서 차이는 있지만 최종 adapter 방향과 크기는 사실상 같다. Eval loss도 1.2035-1.2052, Member exact는 모든 DP 방법에서 0건이다.

## Privacy accounting

| 항목 | 값 |
|---|---:|
| q | 0.0166233766 |
| σ | 1.62109375 |
| Steps | 1,830 |
| δ | 1e-5 |
| Reported PRV ε | 1.9948198277 |
| Independently recomputed PRV ε | 1.9948198277 |
| 최대 오차 | 0 |

`q=128/7700`과 `ceil(7700/128)×30=1830`을 함께 사용하므로 30은 nominal epoch이고 기대 sample pass는 30.4208회다. Privacy accountant는 실제 q와 step을 사용하므로 ε 보장은 이 조건에 대해 정확하다.

## 잔여 위험과 보완

- 실험 속도를 위해 secure RNG를 끌 수 있으므로 production formal claim 전 secure mode 재학습이 필요하다.
- FastDP vendor는 오래된 backward hook을 사용하고 upstream은 구형 PyTorch를 권장한다. 현재 PyTorch 2.10 환경에서는 toy/full 수치 검증을 통과했지만 향후 버전 변경 시 재검증해야 한다.
- FastDP가 unsupported trainable parameter를 조용히 제외하지 못하도록 첫 step에서 364/364 coverage를 강제하는 fail-closed 검사를 추가했다.
- Summary에 sampling/noise seed와 expected sample pass를 기록하도록 보완했다.

## 근거

- [Opacus DPOptimizer](https://opacus.ai/api/optim/dp_optimizer.html)
- [Opacus per-sample gradient guide](https://opacus.ai/tutorials/guide_to_grad_sampler)
- [Opacus Fast/Ghost clipping implementation](https://opacus.ai/api/_modules/opacus/utils/fast_gradient_clipping_utils.html)
- [AWS Labs FastDP](https://github.com/awslabs/fast-differential-privacy)
- Raw numerical result: `dp_backend_equivalence.json`
