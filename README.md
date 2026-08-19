# DP-SGD Method Comparison

Foundation Model의 LoRA fine-tuning 환경에서 DP-SGD 계산 방법을 각자 독립적으로 구현하고, 코드와 실험 결과의 차이를 비교하기 위한 저장소입니다.

## 연구 배경

DP-SGD는 학습 샘플별 gradient를 계산하고 clipping한 뒤 Gaussian noise를 추가하여 학습 데이터에 대한 차분 프라이버시를 제공합니다.

대규모 언어모델에서는 per-sample gradient 계산과 clipping으로 인해 연산량과 GPU 메모리 사용량이 증가합니다. 이 저장소에서는 이러한 계산을 구현하는 여러 방법의 코드 구조, 실행 방식, 학습 결과 및 자원 사용량 차이를 확인합니다.

## 과제 개요

- 과제명: Foundation Model 학습 데이터의 프라이버시 리스크 관리 기술 개발
- 사업: 개인정보 안전활용 선도기술 개발
- 관리체계: 개인정보보호위원회 · 한국인터넷진흥원(KISA)
- 주관기관: 한국전자통신연구원(ETRI)
- 경북대학교 담당: Foundation Model 학습 과정에 적용하는 경량 Differential Privacy 기술

경북대학교는 Foundation Model의 training-time 단계에서 LoRA, 효율적인 DP-SGD 계산, 정밀 Privacy Accountant 및 모델 경량화를 결합하여 개인정보 보호와 학습 효율을 함께 확보하는 연구를 담당합니다.

## 연차별 연구내용

### 1차년도 · LoRA 기반 경량 DP-SGD

- Foundation Model 대상 최신 DP 기술과 공개 구현 조사
- 기존 DP 라이브러리의 대규모 모델 적용 한계 분석
- LoRA 기반 메모리 효율적 DP-SGD 학습 구조 구현
- Vectorized per-example gradient 및 clipping 계산 방법 연구
- 경량 DP-SGD 학습 모듈과 메모리·성능 분석 기반 구축

### 2차년도 · 정밀 Privacy Accountant 및 데이터 중복제거

- RDP 및 Gaussian DP 기반 정밀 Privacy Accountant 고도화
- Epoch, batch size 및 noise 변화에 따른 누적 privacy cost 분석
- Tight privacy budgeting을 통한 불필요한 noise 최소화
- 중복 데이터가 모델 암기와 개인정보 재현에 미치는 영향 분석
- 의미적 유사도 기반 학습 데이터 de-duplication 전처리 기술 개발

### 3차년도 · Pruning, 최적화 및 통합 실증

- 모델 pruning을 적용한 DP 학습·추론 경량화
- Multi-GPU 환경의 DP 연산 및 통신 효율 최적화
- 모델과 데이터셋 규모별 privacy-utility trade-off 검증
- 최적 hyperparameter와 DP 학습 recipe 도출
- 실제 운영 환경 적용을 위한 가이드라인 작성
- ETRI 통합 프라이버시 리스크 관리 프레임워크 연동 및 실증

## 단계별 Privacy 목표

- 1단계(1·2차년도): privacy budget ε ≤ 2.0
- 2단계(3차년도): privacy budget ε ≤ 1.5

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
