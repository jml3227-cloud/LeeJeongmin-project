import numpy as np

def convolution():
    A = np.array([[10, 35, 15], [15, 30, 10], [10, 20, 5]])
    F = np.array([[1, -2], [0, 2]])

    result = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            result[i][j] = np.sum(A[i:i+2, j:j+2] * F)
    
    return result.tolist()

print(convolution())



