# Deep Learning from Scratch

딥러닝 핵심 개념을 외부 라이브러리 없이 직접 구현한 연습 코드 모음

## 폴더 구조

deep_learning/
├── dnn/
│   ├── perceptron.py
│   ├── update_weight.py
│   ├── loss.py
│   ├── sgd_single_variable.py
│   ├── gradient_descent.py
│   ├── gradient_vanishing.py
│   └── baesian.py
├── cnn/
│   └── cnn.py
└── README.md

## 구현 내용

### DNN

**perceptron.py**
- Hard Limit 활성화 함수 구현
- 가중합(Weighted Sum)을 이용한 AND, OR, NOT 게이트 구현
- 단일 퍼셉트론으로는 해결할 수 없는 XOR 문제를 다층 퍼셉트론(MLP)으로 해결

**update_weight.py**
- 계단 함수(Step Function) 기반 퍼셉트론 구현
- 오차(실제값 - 예측값) 계산
- 퍼셉트론 학습 규칙을 이용한 가중치 및 편향 업데이트
  - `새로운 가중치 = 기존 가중치 + (학습률 × 오차 × 입력값)`

**loss.py**
- MAE(Mean Absolute Error), MSE(Mean Squared Error) 구현
- 동일 데이터에 대해 두 손실함수 결과 비교

**sgd_single_variable.py**
- 단변수 경사하강법 수렴 과정 시뮬레이션
- 수렴 조건 구현 (기울기 < 0.001 또는 변화량 < 0.00001)

**gradient_descent.py**
- 경사 하강법 업데이트 공식 구현
  - w(t+1) = w(t) - η × ∂E/∂w

**gradient_vanishing.py**
- Sigmoid 함수 및 도함수 구현
- ReLU 함수 및 도함수 구현
- 역전파 시 활성화 함수 미분값이 누적 곱해지는 과정을 시뮬레이션
  - Sigmoid: 미분값 최대 0.25로 층이 깊어질수록 기울기 소실 발생
  - ReLU: 양수 구간 미분값이 1로 기울기 소실 없음

**baesian.py**
- 베이즈 정리(Bayes' Theorem) 구현
- 사전확률(Prior)과 우도(Likelihood)로 사후확률(Posterior) 계산

### CNN

**cnn.py**
- 2D Convolution 연산 직접 구현
- 슬라이딩 윈도우 방식으로 feature map 생성

---

## 🛠️ 사용 언어
- Python 3.11.9
- 외부 라이브러리 없이 순수 Python으로 구현 (numpy, math 모듈만 사용)