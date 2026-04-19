import math

def sigmoid(z):
    e = math.e
    return 1/(1 + e ** (float(-z)))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
    return max(0, z)

def relu_derivative(z):
    if z > 0:
        return 1
    else:
        return 0

    


print(sigmoid_derivative(0))


def calculate_accumulated_gradient(gradients):
    result = 1
    for i in gradients:
        result *= i
    return result

gradient = [0.25, 0.2, 0.1, 0.15, 0.3]
print(round(calculate_accumulated_gradient(gradient),6))


relu = []
for i in range(1, 6):
    li = relu_derivative(i)
    relu.append(li)

print(relu)
print(calculate_accumulated_gradient(relu))


