input = [4, 8, 12, 16, 20]
target = [6, 5, 16, 12, 20]

def MAE(weight):
    predict = []
    for i in input:
        pred = i * weight
        predict.append(pred)
    
    mae = sum([abs(pi - ti)for pi, ti in zip(predict, target)]) / 5
    return mae

def MSE(weight):
    predict = []
    for i in input:
        pred = i * weight
        predict.append(pred)

    mse = sum([(pi - ti) ** 2 for pi, ti in zip(predict, target)]) / 5
    return mse

print("\n--- 모델 A (y = 9/8x) ---")
print(f"MAE: {MAE(9/8)}")
print(f"MSE: {MSE(9/8)}") 

print("\n--- 모델 B (y = 7/8x) ---")
print(f"MAE: {MAE(7/8)}")
print(f"MSE: {MSE(7/8)}") 
    
    


