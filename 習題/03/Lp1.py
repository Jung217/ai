import random as rd
# 3x + 2y + 5z
# x, y, z >= 0
# x + y <= 10
# 2x + z <= 9
# y + 2z <= 11
# 17/33/11  / 5 = 34.4
cont = 0
while(cont<10000):
    x = y = z = 0
    x = rd.random()*10
    y = rd.random()*10
    z = rd.random()*10
    if x+y<=10 and 2*x+z<=9 and y+2*z<=11:
        if 3*x + 2*y + 5*z > 34.3:
            print(x)
            print(y)
            print(z)
            print(3*x + 2*y + 5*z)
            print()