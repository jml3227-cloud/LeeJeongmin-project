# 단일 퍼셉트론과 기본 논리 게이트(AND, OR, NOT) 구현

def hard_limit(x):
    if x > 0:
        return 1
    else:
        return 0
    

def AND_gate(x1, y1):
    w1 = 1.0
    w2 = 1.0
    w3 = -1.53
    z1 = x1 * w1 + y1 * w2 + w3

    return hard_limit(z1)

def OR_gate(x1, y1):
    w1 = 1.0 
    w2 = 1.0
    w3 = -0.54
    z1 = x1 * w1 + y1 * w2 + w3

    return hard_limit(z1)

def NOT_gate(x1):
    w1 = -1.0
    w2 = 0.5
    z1 = x1 * w1 + w2
    return hard_limit(z1)


def XOR_gate(x1, x2):
    y1 = AND_gate(x1, x2)
    y2 = OR_gate(x1, x2)
    w31 = -1.0
    w32 = 1.0
    w33 = -0.5

    z = w31 * y1 + w32 * y2 + w33
    
    return hard_limit(z)


if __name__ == "__main__":
    print(f"AND(1, 1) = {AND_gate(1,1)}")
    print(f"OR(0, 1) = {OR_gate(0,1)}")
    print(f"NOT(1) = {NOT_gate(1)}")

    print("XOR(0,0) =", XOR_gate(0,0))
    print("XOR(0,1) =", XOR_gate(0,1))
    print("XOR(1,0) =", XOR_gate(1,0))
    print("XOR(1,1) =", XOR_gate(1,1))
