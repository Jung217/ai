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
### 習題四
> 手算反傳遞梯度

![](https://github.com/Jung217/ai/blob/master/%E7%BF%92%E9%A1%8C/04/268811.jpg)
    
### 習題五
> 為 micrograd 加上一個梯度下降法函數 gradientDescendent(...)

```python
#改自 https://github.com/ccc112b/py2cs/blob/master/03-人工智慧/02-優化算法/02-深度學習優化/03-梯度下降法/gd.py
import numpy as np
from numpy.linalg import norm
from micrograd.engine import Value

p = [0.0, 0.0, 0.0]

def gradientDescendent(f, p0, h=0.01, max_loops=100000, dump_period=1000):
    p = p0.copy()
    
    for i in range(max_loops):
        gp, t=[], []
        fp = f(p)

        for j in range(len(p)): t.append(Value(p[j]))
        f(t).backward()

        for j in t: gp.append(j.grad)
        glen = norm(gp) 

        if i%dump_period == 0: print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp, str(p), str(gp), glen))
        if glen < 0.00001: break

        gh = np.multiply(gp, -1*h) 
        p +=  gh 
    print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp, str(p), str(gp), glen))
    return p 

def f(p):
    [x, y, z] = p
    return (x-1)**2+(y-2)**2+(z-3)**2

gradientDescendent(f, p)
```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\05> python .\micro.py
00000:f(p)=14.000 p=[0.0, 0.0, 0.0] gp=[-2.0, -4.0, -6.0] glen=7.48331
00670:f(p)=0.000 p=[0.99999868 1.99999735 2.99999603] gp=[-2.645457017003139e-06, -5.290914034006278e-06, -7.936371051009417e-06] glen=0.00001
```
### 習題六
> 為 macrograd 加上一個 crossEntropyLoss 層，然後用 mnist 測試

```python
```
```
```
### 習題七
> 自己定義一個神經網路模型，並在 MNIST 資料集上訓練並跑出正確率

```py
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):    
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 150, kernel_size=5)
        self.conv2 = nn.Conv2d(150, 10, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\07>  python -W ignore train.py lenet

Test set: Avg. loss: 3.6747, Accuracy: 349/10000 (3%)

Train Epoch: 1 [0/60000 (0%)]   Loss: 3.689767
Train Epoch: 1 [640/60000 (1%)] Loss: 3.261025
Train Epoch: 1 [1280/60000 (2%)]        Loss: 2.698732
Train Epoch: 1 [1920/60000 (3%)]        Loss: 2.081821
Train Epoch: 1 [2560/60000 (4%)]        Loss: 1.877668
Train Epoch: 1 [3200/60000 (5%)]        Loss: 1.402246
Train Epoch: 1 [3840/60000 (6%)]        Loss: 1.292045
Train Epoch: 1 [4480/60000 (7%)]        Loss: 1.247859
Train Epoch: 1 [5120/60000 (9%)]        Loss: 1.340352
Train Epoch: 1 [5760/60000 (10%)]       Loss: 0.903824
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.920512
Train Epoch: 1 [7040/60000 (12%)]       Loss: 1.038622
Train Epoch: 1 [7680/60000 (13%)]       Loss: 0.784540
Train Epoch: 1 [8320/60000 (14%)]       Loss: 0.672007
Train Epoch: 1 [8960/60000 (15%)]       Loss: 0.856007
Train Epoch: 1 [9600/60000 (16%)]       Loss: 0.759974
Train Epoch: 1 [10240/60000 (17%)]      Loss: 0.594563
Train Epoch: 1 [10880/60000 (18%)]      Loss: 0.775489
Train Epoch: 1 [11520/60000 (19%)]      Loss: 0.902855
Train Epoch: 1 [12160/60000 (20%)]      Loss: 0.665745
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.677861
Train Epoch: 1 [13440/60000 (22%)]      Loss: 0.845392
Train Epoch: 1 [14080/60000 (23%)]      Loss: 0.503079
Train Epoch: 1 [14720/60000 (25%)]      Loss: 0.624602
Train Epoch: 1 [15360/60000 (26%)]      Loss: 0.690299
Train Epoch: 1 [16000/60000 (27%)]      Loss: 0.750971
Train Epoch: 1 [16640/60000 (28%)]      Loss: 0.768500
Train Epoch: 1 [17280/60000 (29%)]      Loss: 0.435176
Train Epoch: 1 [17920/60000 (30%)]      Loss: 0.507622
Train Epoch: 1 [18560/60000 (31%)]      Loss: 0.510020
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.381241
Train Epoch: 1 [19840/60000 (33%)]      Loss: 0.351255
Train Epoch: 1 [20480/60000 (34%)]      Loss: 0.657489
Train Epoch: 1 [21120/60000 (35%)]      Loss: 0.654301
Train Epoch: 1 [21760/60000 (36%)]      Loss: 0.380464
Train Epoch: 1 [22400/60000 (37%)]      Loss: 0.351346
Train Epoch: 1 [23040/60000 (38%)]      Loss: 0.364535
Train Epoch: 1 [23680/60000 (39%)]      Loss: 0.459987
Train Epoch: 1 [24320/60000 (41%)]      Loss: 0.437817
Train Epoch: 1 [24960/60000 (42%)]      Loss: 0.398499
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.360488
Train Epoch: 1 [26240/60000 (44%)]      Loss: 0.431571
Train Epoch: 1 [26880/60000 (45%)]      Loss: 0.459572
Train Epoch: 1 [27520/60000 (46%)]      Loss: 0.511598
Train Epoch: 1 [28160/60000 (47%)]      Loss: 0.409274
Train Epoch: 1 [28800/60000 (48%)]      Loss: 0.273368
Train Epoch: 1 [29440/60000 (49%)]      Loss: 0.309932
Train Epoch: 1 [30080/60000 (50%)]      Loss: 0.396242
Train Epoch: 1 [30720/60000 (51%)]      Loss: 0.191774
Train Epoch: 1 [31360/60000 (52%)]      Loss: 0.364342
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.268738
Train Epoch: 1 [32640/60000 (54%)]      Loss: 0.455404
Train Epoch: 1 [33280/60000 (55%)]      Loss: 0.529495
Train Epoch: 1 [33920/60000 (57%)]      Loss: 0.625872
Train Epoch: 1 [34560/60000 (58%)]      Loss: 0.380859
Train Epoch: 1 [35200/60000 (59%)]      Loss: 0.413188
Train Epoch: 1 [35840/60000 (60%)]      Loss: 0.321358
Train Epoch: 1 [36480/60000 (61%)]      Loss: 0.507362
Train Epoch: 1 [37120/60000 (62%)]      Loss: 0.299234
Train Epoch: 1 [37760/60000 (63%)]      Loss: 0.342303
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.295414
Train Epoch: 1 [39040/60000 (65%)]      Loss: 0.481611
Train Epoch: 1 [39680/60000 (66%)]      Loss: 0.351508
Train Epoch: 1 [40320/60000 (67%)]      Loss: 0.155460
Train Epoch: 1 [40960/60000 (68%)]      Loss: 0.228184
Train Epoch: 1 [41600/60000 (69%)]      Loss: 0.215914
Train Epoch: 1 [42240/60000 (70%)]      Loss: 0.200539
Train Epoch: 1 [42880/60000 (71%)]      Loss: 0.212879
Train Epoch: 1 [43520/60000 (72%)]      Loss: 0.336928
Train Epoch: 1 [44160/60000 (74%)]      Loss: 0.349698
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.244631
Train Epoch: 1 [45440/60000 (76%)]      Loss: 0.175509
Train Epoch: 1 [46080/60000 (77%)]      Loss: 0.172243
Train Epoch: 1 [46720/60000 (78%)]      Loss: 0.218962
Train Epoch: 1 [47360/60000 (79%)]      Loss: 0.281713
Train Epoch: 1 [48000/60000 (80%)]      Loss: 0.181091
Train Epoch: 1 [48640/60000 (81%)]      Loss: 0.103181
Train Epoch: 1 [49280/60000 (82%)]      Loss: 0.378462
Train Epoch: 1 [49920/60000 (83%)]      Loss: 0.181984
Train Epoch: 1 [50560/60000 (84%)]      Loss: 0.232347
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.139226
Train Epoch: 1 [51840/60000 (86%)]      Loss: 0.085384
Train Epoch: 1 [52480/60000 (87%)]      Loss: 0.269101
Train Epoch: 1 [53120/60000 (88%)]      Loss: 0.465392
Train Epoch: 1 [53760/60000 (90%)]      Loss: 0.325098
Train Epoch: 1 [54400/60000 (91%)]      Loss: 0.323187
Train Epoch: 1 [55040/60000 (92%)]      Loss: 0.246026
Train Epoch: 1 [55680/60000 (93%)]      Loss: 0.124278
Train Epoch: 1 [56320/60000 (94%)]      Loss: 0.280547
Train Epoch: 1 [56960/60000 (95%)]      Loss: 0.198352
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.278179
Train Epoch: 1 [58240/60000 (97%)]      Loss: 0.144263
Train Epoch: 1 [58880/60000 (98%)]      Loss: 0.333837
Train Epoch: 1 [59520/60000 (99%)]      Loss: 0.208022

Test set: Avg. loss: 0.1159, Accuracy: 9636/10000 (96%)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.280985
Train Epoch: 2 [640/60000 (1%)] Loss: 0.170069
Train Epoch: 2 [1280/60000 (2%)]        Loss: 0.309976
Train Epoch: 2 [1920/60000 (3%)]        Loss: 0.148176
Train Epoch: 2 [2560/60000 (4%)]        Loss: 0.598455
Train Epoch: 2 [3200/60000 (5%)]        Loss: 0.174297
Train Epoch: 2 [3840/60000 (6%)]        Loss: 0.128592
Train Epoch: 2 [4480/60000 (7%)]        Loss: 0.162018
Train Epoch: 2 [5120/60000 (9%)]        Loss: 0.133835
Train Epoch: 2 [5760/60000 (10%)]       Loss: 0.362915
Train Epoch: 2 [6400/60000 (11%)]       Loss: 0.182429
Train Epoch: 2 [7040/60000 (12%)]       Loss: 0.199056
Train Epoch: 2 [7680/60000 (13%)]       Loss: 0.283597
Train Epoch: 2 [8320/60000 (14%)]       Loss: 0.294776
Train Epoch: 2 [8960/60000 (15%)]       Loss: 0.140114
Train Epoch: 2 [9600/60000 (16%)]       Loss: 0.270052
Train Epoch: 2 [10240/60000 (17%)]      Loss: 0.108905
Train Epoch: 2 [10880/60000 (18%)]      Loss: 0.107521
Train Epoch: 2 [11520/60000 (19%)]      Loss: 0.186200
Train Epoch: 2 [12160/60000 (20%)]      Loss: 0.211178
Train Epoch: 2 [12800/60000 (21%)]      Loss: 0.259772
Train Epoch: 2 [13440/60000 (22%)]      Loss: 0.059346
Train Epoch: 2 [14080/60000 (23%)]      Loss: 0.389616
Train Epoch: 2 [14720/60000 (25%)]      Loss: 0.474867
Train Epoch: 2 [15360/60000 (26%)]      Loss: 0.123914
Train Epoch: 2 [16000/60000 (27%)]      Loss: 0.151233
Train Epoch: 2 [16640/60000 (28%)]      Loss: 0.314821
Train Epoch: 2 [17280/60000 (29%)]      Loss: 0.263271
Train Epoch: 2 [17920/60000 (30%)]      Loss: 0.092592
Train Epoch: 2 [18560/60000 (31%)]      Loss: 0.150811
Train Epoch: 2 [19200/60000 (32%)]      Loss: 0.102119
Train Epoch: 2 [19840/60000 (33%)]      Loss: 0.107836
Train Epoch: 2 [20480/60000 (34%)]      Loss: 0.117738
Train Epoch: 2 [21120/60000 (35%)]      Loss: 0.136240
Train Epoch: 2 [21760/60000 (36%)]      Loss: 0.192458
Train Epoch: 2 [22400/60000 (37%)]      Loss: 0.237982
Train Epoch: 2 [23040/60000 (38%)]      Loss: 0.109439
Train Epoch: 2 [23680/60000 (39%)]      Loss: 0.097123
Train Epoch: 2 [24320/60000 (41%)]      Loss: 0.123059
Train Epoch: 2 [24960/60000 (42%)]      Loss: 0.188623
Train Epoch: 2 [25600/60000 (43%)]      Loss: 0.108611
Train Epoch: 2 [26240/60000 (44%)]      Loss: 0.295738
Train Epoch: 2 [26880/60000 (45%)]      Loss: 0.223088
Train Epoch: 2 [27520/60000 (46%)]      Loss: 0.136467
Train Epoch: 2 [28160/60000 (47%)]      Loss: 0.082943
Train Epoch: 2 [28800/60000 (48%)]      Loss: 0.183123
Train Epoch: 2 [29440/60000 (49%)]      Loss: 0.250069
Train Epoch: 2 [30080/60000 (50%)]      Loss: 0.178686
Train Epoch: 2 [30720/60000 (51%)]      Loss: 0.314156
Train Epoch: 2 [31360/60000 (52%)]      Loss: 0.258203
Train Epoch: 2 [32000/60000 (53%)]      Loss: 0.194244
Train Epoch: 2 [32640/60000 (54%)]      Loss: 0.206889
Train Epoch: 2 [33280/60000 (55%)]      Loss: 0.192257
Train Epoch: 2 [33920/60000 (57%)]      Loss: 0.148743
Train Epoch: 2 [34560/60000 (58%)]      Loss: 0.103903
Train Epoch: 2 [35200/60000 (59%)]      Loss: 0.182443
Train Epoch: 2 [35840/60000 (60%)]      Loss: 0.426481
Train Epoch: 2 [36480/60000 (61%)]      Loss: 0.110395
Train Epoch: 2 [37120/60000 (62%)]      Loss: 0.121436
Train Epoch: 2 [37760/60000 (63%)]      Loss: 0.125824
Train Epoch: 2 [38400/60000 (64%)]      Loss: 0.124697
Train Epoch: 2 [39040/60000 (65%)]      Loss: 0.157802
Train Epoch: 2 [39680/60000 (66%)]      Loss: 0.216388
Train Epoch: 2 [40320/60000 (67%)]      Loss: 0.160630
Train Epoch: 2 [40960/60000 (68%)]      Loss: 0.767837
Train Epoch: 2 [41600/60000 (69%)]      Loss: 0.156023
Train Epoch: 2 [42240/60000 (70%)]      Loss: 0.204863
Train Epoch: 2 [42880/60000 (71%)]      Loss: 0.098066
Train Epoch: 2 [43520/60000 (72%)]      Loss: 0.173227
Train Epoch: 2 [44160/60000 (74%)]      Loss: 0.163504
Train Epoch: 2 [44800/60000 (75%)]      Loss: 0.061704
Train Epoch: 2 [45440/60000 (76%)]      Loss: 0.198469
Train Epoch: 2 [46080/60000 (77%)]      Loss: 0.397366
Train Epoch: 2 [46720/60000 (78%)]      Loss: 0.185322
Train Epoch: 2 [47360/60000 (79%)]      Loss: 0.075444
Train Epoch: 2 [48000/60000 (80%)]      Loss: 0.106482
Train Epoch: 2 [48640/60000 (81%)]      Loss: 0.234607
Train Epoch: 2 [49280/60000 (82%)]      Loss: 0.213526
Train Epoch: 2 [49920/60000 (83%)]      Loss: 0.069280
Train Epoch: 2 [50560/60000 (84%)]      Loss: 0.109612
Train Epoch: 2 [51200/60000 (85%)]      Loss: 0.112785
Train Epoch: 2 [51840/60000 (86%)]      Loss: 0.164431
Train Epoch: 2 [52480/60000 (87%)]      Loss: 0.097833
Train Epoch: 2 [53120/60000 (88%)]      Loss: 0.179836
Train Epoch: 2 [53760/60000 (90%)]      Loss: 0.174302
Train Epoch: 2 [54400/60000 (91%)]      Loss: 0.221067
Train Epoch: 2 [55040/60000 (92%)]      Loss: 0.234614
Train Epoch: 2 [55680/60000 (93%)]      Loss: 0.129750
Train Epoch: 2 [56320/60000 (94%)]      Loss: 0.121351
Train Epoch: 2 [56960/60000 (95%)]      Loss: 0.113052
Train Epoch: 2 [57600/60000 (96%)]      Loss: 0.203647
Train Epoch: 2 [58240/60000 (97%)]      Loss: 0.451691
Train Epoch: 2 [58880/60000 (98%)]      Loss: 0.263307
Train Epoch: 2 [59520/60000 (99%)]      Loss: 0.138063

Test set: Avg. loss: 0.0846, Accuracy: 9725/10000 (97%)

Train Epoch: 3 [0/60000 (0%)]   Loss: 0.139691
Train Epoch: 3 [640/60000 (1%)] Loss: 0.089846
Train Epoch: 3 [1280/60000 (2%)]        Loss: 0.261636
Train Epoch: 3 [1920/60000 (3%)]        Loss: 0.272293
Train Epoch: 3 [2560/60000 (4%)]        Loss: 0.185451
Train Epoch: 3 [3200/60000 (5%)]        Loss: 0.055740
Train Epoch: 3 [3840/60000 (6%)]        Loss: 0.121596
Train Epoch: 3 [4480/60000 (7%)]        Loss: 0.186014
Train Epoch: 3 [5120/60000 (9%)]        Loss: 0.157091
Train Epoch: 3 [5760/60000 (10%)]       Loss: 0.169451
Train Epoch: 3 [6400/60000 (11%)]       Loss: 0.200788
Train Epoch: 3 [7040/60000 (12%)]       Loss: 0.141811
Train Epoch: 3 [7680/60000 (13%)]       Loss: 0.238299
Train Epoch: 3 [8320/60000 (14%)]       Loss: 0.096617
Train Epoch: 3 [8960/60000 (15%)]       Loss: 0.157436
Train Epoch: 3 [9600/60000 (16%)]       Loss: 0.081993
Train Epoch: 3 [10240/60000 (17%)]      Loss: 0.244759
Train Epoch: 3 [10880/60000 (18%)]      Loss: 0.029848
Train Epoch: 3 [11520/60000 (19%)]      Loss: 0.307981
Train Epoch: 3 [12160/60000 (20%)]      Loss: 0.127715
Train Epoch: 3 [12800/60000 (21%)]      Loss: 0.210392
Train Epoch: 3 [13440/60000 (22%)]      Loss: 0.112803
Train Epoch: 3 [14080/60000 (23%)]      Loss: 0.342360
Train Epoch: 3 [14720/60000 (25%)]      Loss: 0.204126
Train Epoch: 3 [15360/60000 (26%)]      Loss: 0.235766
Train Epoch: 3 [16000/60000 (27%)]      Loss: 0.174971
Train Epoch: 3 [16640/60000 (28%)]      Loss: 0.112329
Train Epoch: 3 [17280/60000 (29%)]      Loss: 0.268389
Train Epoch: 3 [17920/60000 (30%)]      Loss: 0.067175
Train Epoch: 3 [18560/60000 (31%)]      Loss: 0.099363
Train Epoch: 3 [19200/60000 (32%)]      Loss: 0.173430
Train Epoch: 3 [19840/60000 (33%)]      Loss: 0.087307
Train Epoch: 3 [20480/60000 (34%)]      Loss: 0.136345
Train Epoch: 3 [21120/60000 (35%)]      Loss: 0.202613
Train Epoch: 3 [21760/60000 (36%)]      Loss: 0.127024
Train Epoch: 3 [22400/60000 (37%)]      Loss: 0.240312
Train Epoch: 3 [23040/60000 (38%)]      Loss: 0.079366
Train Epoch: 3 [23680/60000 (39%)]      Loss: 0.113079
Train Epoch: 3 [24320/60000 (41%)]      Loss: 0.118096
Train Epoch: 3 [24960/60000 (42%)]      Loss: 0.115985
Train Epoch: 3 [25600/60000 (43%)]      Loss: 0.102411
Train Epoch: 3 [26240/60000 (44%)]      Loss: 0.190959
Train Epoch: 3 [26880/60000 (45%)]      Loss: 0.125113
Train Epoch: 3 [27520/60000 (46%)]      Loss: 0.153628
Train Epoch: 3 [28160/60000 (47%)]      Loss: 0.188062
Train Epoch: 3 [28800/60000 (48%)]      Loss: 0.104512
Train Epoch: 3 [29440/60000 (49%)]      Loss: 0.165989
Train Epoch: 3 [30080/60000 (50%)]      Loss: 0.118717
Train Epoch: 3 [30720/60000 (51%)]      Loss: 0.113525
Train Epoch: 3 [31360/60000 (52%)]      Loss: 0.167668
Train Epoch: 3 [32000/60000 (53%)]      Loss: 0.189769
Train Epoch: 3 [32640/60000 (54%)]      Loss: 0.087908
Train Epoch: 3 [33280/60000 (55%)]      Loss: 0.271235
Train Epoch: 3 [33920/60000 (57%)]      Loss: 0.224298
Train Epoch: 3 [34560/60000 (58%)]      Loss: 0.342680
Train Epoch: 3 [35200/60000 (59%)]      Loss: 0.164589
Train Epoch: 3 [35840/60000 (60%)]      Loss: 0.217280
Train Epoch: 3 [36480/60000 (61%)]      Loss: 0.132709
Train Epoch: 3 [37120/60000 (62%)]      Loss: 0.279049
Train Epoch: 3 [37760/60000 (63%)]      Loss: 0.047063
Train Epoch: 3 [38400/60000 (64%)]      Loss: 0.101154
Train Epoch: 3 [39040/60000 (65%)]      Loss: 0.126646
Train Epoch: 3 [39680/60000 (66%)]      Loss: 0.463659
Train Epoch: 3 [40320/60000 (67%)]      Loss: 0.308736
Train Epoch: 3 [40960/60000 (68%)]      Loss: 0.074244
Train Epoch: 3 [41600/60000 (69%)]      Loss: 0.218058
Train Epoch: 3 [42240/60000 (70%)]      Loss: 0.312545
Train Epoch: 3 [42880/60000 (71%)]      Loss: 0.332677
Train Epoch: 3 [43520/60000 (72%)]      Loss: 0.169871
Train Epoch: 3 [44160/60000 (74%)]      Loss: 0.145763
Train Epoch: 3 [44800/60000 (75%)]      Loss: 0.216663
Train Epoch: 3 [45440/60000 (76%)]      Loss: 0.213525
Train Epoch: 3 [46080/60000 (77%)]      Loss: 0.078893
Train Epoch: 3 [46720/60000 (78%)]      Loss: 0.075514
Train Epoch: 3 [47360/60000 (79%)]      Loss: 0.197619
Train Epoch: 3 [48000/60000 (80%)]      Loss: 0.076122
Train Epoch: 3 [48640/60000 (81%)]      Loss: 0.103198
Train Epoch: 3 [49280/60000 (82%)]      Loss: 0.132155
Train Epoch: 3 [49920/60000 (83%)]      Loss: 0.055669
Train Epoch: 3 [50560/60000 (84%)]      Loss: 0.094455
Train Epoch: 3 [51200/60000 (85%)]      Loss: 0.058275
Train Epoch: 3 [51840/60000 (86%)]      Loss: 0.087515
Train Epoch: 3 [52480/60000 (87%)]      Loss: 0.125367
Train Epoch: 3 [53120/60000 (88%)]      Loss: 0.074382
Train Epoch: 3 [53760/60000 (90%)]      Loss: 0.151455
Train Epoch: 3 [54400/60000 (91%)]      Loss: 0.177782
Train Epoch: 3 [55040/60000 (92%)]      Loss: 0.158581
Train Epoch: 3 [55680/60000 (93%)]      Loss: 0.112333
Train Epoch: 3 [56320/60000 (94%)]      Loss: 0.086345
Train Epoch: 3 [56960/60000 (95%)]      Loss: 0.130908
Train Epoch: 3 [57600/60000 (96%)]      Loss: 0.125547
Train Epoch: 3 [58240/60000 (97%)]      Loss: 0.246290
Train Epoch: 3 [58880/60000 (98%)]      Loss: 0.283334
Train Epoch: 3 [59520/60000 (99%)]      Loss: 0.105566

Test set: Avg. loss: 0.0637, Accuracy: 9791/10000 (98%)
```
### 習題八
> 自己設計一個固定的策略（不需要學習）解決 CartPole 問題，讓你的竿子盡量撐得久不會倒下來

```py
# 參考範例並原創
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

steps = 0
max_steps = 0

for _ in range(1000):
    env.render()

    position, velocity, angle, angular_velocity = observation

    if angle > 0:
        if angular_velocity > 0: action = 1
        else: action = 0 
    else:
        if angular_velocity < 0: action = 0 
        else: action = 1 

    if position > 0.1: action = 0
    elif position < -0.1: action = 1

    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if terminated or truncated:
        if steps > max_steps: max_steps = steps
        print(f'Died after {steps} steps')
        steps = 0
        observation, info = env.reset()

env.close()
print(f'Maximum steps survived: {max_steps}')
```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\08> python .\cartpole.py
Died after 57 steps
Died after 116 steps
Died after 102 steps
Died after 131 steps
Died after 88 steps
Died after 46 steps
Died after 131 steps
Died after 73 steps
Died after 180 steps
Maximum steps survived: 180
```
### 習題九
> 請呼叫 LLM 大語言模型 api (groq, openai) 去做一個小應用

* [Jung217 / groq_line_bot](https://github.com/Jung217/groq_line_bot)
### 習題十
> 

```py
```
```
```
### 期中
* [人工智慧的筆記](https://hackmd.io/@Jung217/nqu_ai)
* [基於 LSTM & BERT 機器學習之網路輿情分析](https://github.com/Jung217/LSTM_BERT_Sentiment_Analysis)