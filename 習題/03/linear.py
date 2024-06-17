# 參考 https://github.com/wrr606/ai/blob/master/homework/3/liner.py 的程式脈絡
import random as rd

# 3x + 2y + 5z
# x, y, z >= 0
# x + y <= 10
# 2x + z <= 9
# y + 2z <= 11

def fn(x, y, z):
    return 3*x + 2*y + 5*z

def limit(x, y, z):
    return (x + y <= 10) and (2*x + z <= 9) and (y + 2*z <= 11) and (x >= 0) and (y >= 0) and (z >= 0)

def neighbor(x, y, z):
    step = 0.1 
    direction = rd.choice(['x', 'y', 'z'])
    
    if direction == 'x': x += rd.choice([-step, step])
    elif direction == 'y': y += rd.choice([-step, step])
    elif direction == 'z': z += rd.choice([-step, step])

    return x, y, z

def hillClimbing():
    fail = 0

    x, y, z = rd.uniform(0, 10), rd.uniform(0, 10), rd.uniform(0, 10)
    while limit(x, y, z) != True: x, y, z = rd.uniform(0, 10), rd.uniform(0, 10), rd.uniform(0, 10)
    
    curvalue = fn(x, y, z)

    while fail < 100000:
        newX, newY, newZ = neighbor(x, y, z)
        newvalue = fn(newX, newY, newZ)

        if limit(newX, newY, newZ) and newvalue > curvalue:
            x, y, z = newX, newY, newZ
            curvalue = newvalue
            fail = 0
        else: fail += 1
    
    return x, y, z, curvalue

best_x, best_y, best_z, best_value = hillClimbing()
print(f"Optimal solution: x = {best_x:.2f}, y = {best_y:.2f}, z = {best_z:.2f}")
print(f"Maximum value of the objective function: {best_value:.2f}")