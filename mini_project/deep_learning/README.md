## 📌 챕터별 설명

### ch01. 단일 퍼셉트론과 논리 게이트 구현 (`perceptron.py`)

- Hard Limit 활성화 함수 구현
- 가중합(Weighted Sum)을 이용한 AND, OR, NOT 게이트 구현
- 단일 퍼셉트론으로는 해결할 수 없는 XOR 문제를 다층 퍼셉트론(MLP)으로 해결
  - 은닉층에 AND, OR 게이트를 조합하여 비선형 분리 구현

### ch02. 신경망 학습과 가중치 업데이트 (`updateweight.py`)

- 계단 함수(Step Function) 기반 퍼셉트론 구현
- 오차(실제값 - 예측값) 계산
- 퍼셉트론 학습 규칙을 이용한 가중치 및 편향 업데이트
  - `새로운 가중치 = 기존 가중치 + (학습률 × 오차 × 입력값)`

### ch03. Gradient Vanishing 시뮬레이션 (`gradientvanishing.py`)

- Sigmoid 함수 및 도함수 구현
- ReLU 함수 및 도함수 구현
- 역전파 시 활성화 함수 미분값이 누적 곱해지는 과정을 시뮬레이션
  - Sigmoid: 미분값 최대 0.25로 층이 깊어질수록 기울기 소실 발생
  - ReLU: 양수 구간 미분값이 1로 기울기 소실 없음

### ch04. 경사 하강법 가중치 업데이트 (`gradient_descent.py`)

- 경사 하강법 업데이트 공식 구현
  - `w(t+1) = w(t) - η × ∂E/∂w`

---

## ▶️ 실행 방법

```bash
python ch01_perceptron/perceptron.py
python ch02_weight_update/updateweight.py
python ch03_gradient_vanishing/gradientvanishing.py
python ch04_gradient_descent/gradient_descent.py
```

---

## 🛠️ 사용 언어

- Python 3.11.9
- 외부 라이브러리 없이 순수 Python으로 구현 (math 모듈만 사용)
