# DP-SGD Method Comparison

Foundation Model의 LoRA fine-tuning 환경에서 DP-SGD 계산 방법을 각자 독립적으로 구현하고, 코드와 실험 결과의 차이를 비교하기 위한 저장소입니다.

## 연구 배경

DP-SGD는 학습 샘플별 gradient를 계산하고 clipping한 뒤 Gaussian noise를 추가하여 학습 데이터에 대한 차분 프라이버시를 제공합니다.

대규모 언어모델에서는 per-sample gradient 계산과 clipping으로 인해 연산량과 GPU 메모리 사용량이 증가합니다. 이 저장소에서는 이러한 계산을 구현하는 여러 방법의 코드 구조, 실행 방식, 학습 결과 및 자원 사용량 차이를 확인합니다.

## 검토 대상

- Opacus Hooks
- Functorch 기반 per-sample gradient
- ExpandedWeights
- Ghost Clipping
- FastDP Book-Keeping
- 기타 DP-SGD 구현 방식

## 저장소 운영 방식

각 참여자는 자신의 이름으로 폴더를 만들고, 사용한 코드와 설정 및 결과를 해당 폴더 안에 독립적으로 기록합니다.

메인 저장소에서는 특정 실험 조건이나 구현 방식을 정답으로 지정하지 않습니다. 비교 시에는 각 참여자 폴더에 기록된 실제 코드와 설정을 기준으로 차이를 확인합니다.

```text
dp-sgd-method/
├── README.md
├── participant-a/
│   ├── README.md
│   ├── code/
│   ├── configs/
│   └── results/
└── participant-b/
    ├── README.md
    ├── code/
    ├── configs/
    └── results/
```

각 참여자 폴더에는 다음 내용을 포함할 수 있습니다.

- 학습 및 평가 코드
- 실제 실행에 사용한 설정 파일
- 실행 환경과 라이브러리 버전
- 실험 결과와 로그 요약
- 구현 과정에서 확인한 제약과 차이점

모델 checkpoint, 데이터셋 원본 및 기타 대용량 파일은 저장소에 직접 추가하지 않습니다.
