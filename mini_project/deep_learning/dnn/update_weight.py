def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


x = [90, 60, 80, 80]
w = [0.1, 0.2, 0.3, 0.1]
b = -20

def perceptron(x, w, b):
    z = sum(xi * wi for xi, wi in zip(x, w)) + b
    return step_function(z)

def error(x, w, b):
    y_pred = perceptron(x, w, b)
    return 0 - y_pred

def update(x, w, b):
    lr = 0.005
    err = error(x, w, b)
    for i in range(len(w)):
        w[i] = w[i] + (lr * err * x[i])

    b = b + (lr * err * 1)
    return w, b

print(f"초기 예측 결과 : {perceptron(x, w, b)}")
print("실제 결과 : 불합격(0)")
print(f"발생한 오차 : {error(x, w, b)}")

w, b = update(x, w, b)

print("\n--- 가중치 및 편향 업데이트 ---")
print(f"업데이트된 국어 가중치 : {round(w[0], 2)}")
print(f"업데이트된 수학 가중치 : {round(w[1], 2)}")
print(f"업데이트된 영어 가중치 : {round(w[2], 3)}")
print(f"업데이트된 과학 가중치 : {round(w[3], 3)}")
print(f"업데이트된 편향 : {round(b, 3)}")