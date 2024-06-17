# 改自老師上課範例
from random import randint

citys = [
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
]

l = len(citys)
path = [(i+1)%l for i in range(l)]
print('Initial path:', path)

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def pathLength(p):
    dist = 0
    plen = len(p)
    for i in range(plen):
        dist += distance(citys[p[i]], citys[p[(i+1)%plen]])
    return dist

print('Initial pathLength=', pathLength(path))

class Solution:
    def __init__(self, v):
        self.v = v

    def neighbor(self): # 隨機換兩個城市
        fills = self.v.copy()
        i = randint(0, len(fills)-1)
        j = randint(0, len(fills)-1)
        fills[i], fills[j] = fills[j], fills[i]
        return Solution(fills)

    def height(self): # 高度（因為爬山找的最大值 所以負用的路徑長度）
        return -pathLength(self.v)

    def __str__(self):
        return 'pathLength={:f} path={}'.format(pathLength(self.v), self.v)

def hillClimbing(x, max_fail=1000):
    fail = 0
    while True:
        nx = x.neighbor()
        if nx.height() > x.height():
            x = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return x

initial_solution = Solution(path)
solution = hillClimbing(initial_solution)
print("INITAL:", initial_solution)
print("Final:", solution)
