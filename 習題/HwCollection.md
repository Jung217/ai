## 習題
> 各題原創或參考均標於程式開頭
### 習題一
> 請寫一個排課系統

```python
# 參考老師上課範例
from random import randint

# teachers = ['甲', '乙', '丙', '丁', '戊']
# rooms = ['A', 'B']
# cols = 7

courses = [
{'teacher': '  ', 'name':'　　', 'hours': -1},
{'teacher': '甲', 'name':'機率', 'hours': 2},
{'teacher': '甲', 'name':'線代', 'hours': 3},
{'teacher': '甲', 'name':'離散', 'hours': 3},
{'teacher': '乙', 'name':'視窗', 'hours': 3},
{'teacher': '乙', 'name':'科學', 'hours': 3},
{'teacher': '乙', 'name':'系統', 'hours': 3},
{'teacher': '乙', 'name':'計概', 'hours': 3},
{'teacher': '丙', 'name':'軟工', 'hours': 3},
{'teacher': '丙', 'name':'行動', 'hours': 3},
{'teacher': '丙', 'name':'網路', 'hours': 3},
{'teacher': '丁', 'name':'媒體', 'hours': 3},
{'teacher': '丁', 'name':'工數', 'hours': 3},
{'teacher': '丁', 'name':'動畫', 'hours': 3},
{'teacher': '丁', 'name':'電子', 'hours': 4},
{'teacher': '丁', 'name':'嵌入', 'hours': 3},
{'teacher': '戊', 'name':'網站', 'hours': 3},
{'teacher': '戊', 'name':'網頁', 'hours': 3},
{'teacher': '戊', 'name':'演算', 'hours': 3},
{'teacher': '戊', 'name':'結構', 'hours': 3},
{'teacher': '戊', 'name':'智慧', 'hours': 3}
]

slots = [
'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17',
'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27',
'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37',
'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47',
'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57',
]

class Solution: # 解答的物件模版 (類別)
    def __init__(self, v, step = 0.01):
        self.v = v       # 參數 v 為解答的資料結構
        self.step = step # 每一小步預設走的距離

    # 以下兩個函數至少需要覆蓋掉一個，否則會無窮遞迴
    def height(self): # 爬山演算法的高度函數
        return -1*self.energy()               # 高度 = -1 * 能量

    def energy(self): # 尋找最低點的能量函數
        return -1*self.height()               # 能量 = -1 * 高度

def randSlot() :
    return randint(0, len(slots)-1)

def randCourse() :
    return randint(0, len(courses)-1)


class SolutionScheduling(Solution) :
    def neighbor(self): # 單變數解答的鄰居函數。
        fills = self.v.copy()
        choose = randint(0, 1)
        if choose == 0: # 任選一個改變 
            i = randSlot()
            fills[i] = randCourse()
        elif choose == 1: # 任選兩個交換
            i = randSlot()
            j = randSlot()
            t = fills[i]
            fills[i] = fills[j]
            fills[j] = t
        return SolutionScheduling(fills) # 建立新解答並傳回。

    def height(self): # 高度函數
        courseCounts = [0] * len(courses)
        fills = self.v
        score = 0
        for si in range(len(slots)):
            courseCounts[fills[si]] += 1
            if si < len(slots)-1 and fills[si] == fills[si+1] and si%7 != 6 and si%7 != 3: #連續上課:好；隔天:不好；跨越中午:不好
                score += 0.12
            if si % 7 == 0 and fills[si] != 0: # 早上 8:00: 不好
                score -= 0.19
        for ci in range(len(courses)):
            if (courses[ci]['hours'] >= 0):
                score -= abs(courseCounts[ci] - courses[ci]['hours']) # 課程總時數不對: 不好
        return score

    def __str__(self): # 將解答轉為字串，以供印出觀察。
        outs = []
        fills = self.v
        for i in range(len(slots)):
            c = courses[fills[i]]
            if i%7 == 0:
                outs.append('\n')
            outs.append(slots[i] + ':' + c['name'])
        return 'height={:f} {:s}\n\n'.format(self.height(), ' '.join(outs))
    
    @classmethod
    def init(cls):
        fills = [0] * len(slots)
        for i in range(len(slots)):
            fills[i] = randCourse()
        return SolutionScheduling(fills)
    

def hillClimbing(x, max_fail=1000): # 通用爬山演算法框架
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

solution = hillClimbing(SolutionScheduling.init())
print("Final :", solution)
```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\01> python .\course_scheduling.py
Final : height=3.390000 
 A11:機率 A12:機率 A13:計概 A14:計概 A15:計概 A16:軟工 A17:軟工
 A21:　　 A22:網站 A23:網站 A24:網站 A25:網路 A26:網路 A27:網路
 A31:　　 A32:動畫 A33:動畫 A34:動畫 A35:行動 A36:行動 A37:網頁
 A41:　　 A42:離散 A43:離散 A44:演算 A45:結構 A46:結構 A47:結構
 A51:　　 A52:線代 A53:線代 A54:線代 A55:系統 A56:系統 A57:系統
 B11:媒體 B12:媒體 B13:智慧 B14:智慧 B15:視窗 B16:視窗 B17:視窗
 B21:工數 B22:工數 B23:工數 B24:媒體 B25:科學 B26:科學 B27:科學
 B31:　　 B32:　　 B33:嵌入 B34:嵌入 B35:智慧 B36:電子 B37:電子
 B41:　　 B42:　　 B43:網頁 B44:網頁 B45:行動 B46:電子 B47:電子
 B51:　　 B52:　　 B53:演算 B54:演算 B55:離散 B56:軟工 B57:嵌入
```
### 習題二
> 請用爬山演算法解決 旅行推銷員問題

```python
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

```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\02> python .\tsp.py
Initial path: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
Initial pathLength= 28.90104654287823
INITAL: pathLength=28.901047 path=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
Final: pathLength=12.000000 path=[0, 2, 3, 1, 4, 6, 8, 10, 11, 9, 7, 5]
```
### 習題三
> 請寫一個程式解決線性規劃問題 
```python
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
```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\03> python .\linear.py
Optimal solution: x = 2.98, y = 4.86, z = 3.03
Maximum value of the objective function: 33.79
```
### 習題四 : 求解方程式
> 
![](https://github.com/Jung217/ai/blob/master/%E7%BF%92%E9%A1%8C/04/268811.jpg)
    
### 習題五:找任何向量函數的最高點
> 

```python
```
```
```
### 習題六
> 

```python
```
```
```
### 習題七
> 

```py
```
```
```
### 習題八
> 

```py
```
```
```
### 習題九
> 

```py
```
```
```
### 習題十
> 

```py
```
```
```
