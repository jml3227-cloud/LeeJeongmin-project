def update_weight(current_w, eta, grad):
    return current_w - eta * grad

print(f"업데이트된 가중치: {update_weight(0.5, 0.01, 2.0)}")