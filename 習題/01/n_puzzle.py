originP = [[15, 5, 2, 4],
           [3, 13, 7, 8],
           [1, 10, 11, 12],
           [6, 14, 9, 0]]

originP = [15, 5, 2, 4, 3, 13, 7, 8, 1, 10, 11, 12, 6, 14, 0, 9]
targetP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15, 0]

def height(array):
    arrayP = array.copy()
    H = 0
    for i in targetP:
        if arrayP[i] == targetP[i]: H += 1
    return H

def neighbor(array):
    for i in range(0, len(array)-1):
        if array[i] == 0: 
            array[i] = array[i+1]
            array[i+1] = 0
        elif array[i] > array[i+1]:
            temp = array[i]
            array[i] = array[i+1]
            array[i+1] = temp
    return array

def hillClimbing(x, max_fail=10000):
    fail = 0
    while True:
        nx = neighbor(originP)
        if height(nx) > height(x):
            print(nx)
            x = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail: return x

#print(hillClimbing(originP))
print(len(originP))            
print(neighbor(originP))
            