def loss(a):
    return 12.5 * a -15

def gradient_descent(a0 = 5, r = 0.06, epochs=10000):
    a = a0

    print("반복(n)\ta_n\t\tL'(a_n)\t\ta_{n+1}")

    for i in range(epochs):
        new_a = a - r * loss(a)

        print(f"{i}\t{a:<15.6f}\t{loss(a):<15.6f}\t{new_a:.6f}")

        if abs(new_a - a) < 0.00001 or abs(loss(a)) < 0.001:
            break

        a = new_a

    print(f"[종료] 접선의 기울기가 0에 수렴했습니다. (반복 횟수: {i+1})")
    print(f"최종 최적화된 a값: {a:.6f} (이론적 최솟값 1.2에 근접)")

gradient_descent()