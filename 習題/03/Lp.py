# 3x + 2y + 5z
# x, y, z >= 0
# x + y <= 10
# 2x + z <= 9
# y + 2z <= 11
# 17/33/11  / 5 = 34.4
x = y = z = 0
a1 = [1, 1, 0, 10]
a2 = [2, 0, 1, 9]
a3 = [0, 1, 2, 11]
a4 = [0, 0, 0, 0]
a5 = [0, 0, 0, 0]

for i in range(0, len(a4)):
    a4[i] = 2*a1[i] - a2[i]
    a5[i] = a4[i] - 2*a3[i]
z = a5[3] / a5[2]
x = (9-z) / 2
y = 11 - 2*z
print(x)
print(y)
print(z)
print(3*x+2*y+5*z)