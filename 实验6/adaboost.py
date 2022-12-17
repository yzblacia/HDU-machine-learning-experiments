import math

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
# 初始化权值分布
w0 = [0.1 for i in range(0, 10)]

# first split point 2.5
F1 = [1 if xi < 2.5 else -1 for xi in x]
e1 = [w0[i] if F1[i] != y[i] else 0 for i in range(0, 10)]
e1 = sum(e1)  # 错误率
print(e1)
a1 = 0.5 * math.log((1 - e1) / e1)
Z = [w0[i] * math.exp(-a1 * F1[i] * y[i]) for i in range(0, 10)]
Zs = sum(Z)  # 归一化常数
# print(Zs)
# print(2*math.sqrt(e1*(1-e1)))
w1 = [z / Zs for z in Z]  # 根椐错误率调整后的权值分布
print(w1)

# second split point 8.5
F2 = [1 if xi < 8.5 else -1 for xi in x]
e2 = [w1[i] if F2[i] != y[i] else 0 for i in range(0, 10)]
e2 = sum(e2)
print(e2)
a2 = 0.5 * math.log((1 - e2) / e2)
Z = [w1[i] * math.exp(-a2 * F2[i] * y[i]) for i in range(0, 10)]
Zs = sum(Z)
w2 = [z / Zs for z in Z]
print(w2)

# third split point 5.5
F3 = [-1 if xi < 5.5 else 1 for xi in x]
e3 = [w2[i] if F3[i] != y[i] else 0 for i in range(0, 10)]
e3 = sum(e3)
print(e3)
a3 = 0.5 * math.log((1 - e3) / e3)
Z = [w2[i] * math.exp(-a3 * F3[i] * y[i]) for i in range(0, 10)]
Zs = sum(Z)
w3 = [z / Zs for z in Z]
print(w3)
# final split function
G3 = [a1 * F1[i] + a2 * F2[i] + a3 * F3[i] for i in range(0, 10)]
G3 = [1 if g > 0 else -1 for g in G3]
print(G3)
