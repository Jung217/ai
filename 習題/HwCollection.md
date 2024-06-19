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
# 參考 https://github.com/ccc112b/py2gpt/blob/master/03b-MacroGrad/macrograd/engine.py
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    @property
    def shape(self):
        return self.data.shape
    
    def __add__(self, other):
        # assert self.shape == other.shape
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other) # 讓維度一致
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # print('self.grad = ', self.grad)
            # print('other.grad = ', other.grad)
            # print('out.grad = ', out.grad, 'op=', out._op)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other) # 讓維度一致
        # other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # print('self.shape=', self.shape)
            # print('other.shape=', other.shape)
            # print('out.shape=', out.shape)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
                        
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu') # Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def matmul(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data , other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.dot(out.grad,other.data.T)
            other.grad += np.dot(self.data.T,out.grad)            
            
        out._backward = _backward

        return out

    def softmax(self):
        out =  Tensor(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
        softmax = out.data

        def _backward():
            s = np.sum(out.grad * softmax, 1)
            t = np.reshape(s, [-1, 1]) # reshape 為 n*1
            self.grad += (out.grad - t) * softmax

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data),(self,),'log')

        def _backward():
            self.grad += out.grad/self.data
        out._backward = _backward

        return out    
    
    def sum(self,axis = None):
        out = Tensor(np.sum(self.data,axis = axis), (self,), 'SUM')
        
        def _backward():
            output_shape = np.array(self.data.shape)
            output_shape[axis] = 1
            tile_scaling = self.data.shape // output_shape
            grad = np.reshape(out.grad, output_shape)
            self.grad += np.tile(grad, tile_scaling)
            
        out._backward = _backward

        return out

    def cross_entropy(self, yb): 
        epsilon = 1e-12 
        ce = -(yb * np.log(self.data + epsilon)).sum(axis=-1).sum()  
        return ce

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            #print(v)
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
```
```python
# 參考 https://github.com/ccc112b/py2gpt/blob/master/03b-MacroGrad/macrograd/engine.py
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    @property
    def shape(self):
        return self.data.shape
    
    def __add__(self, other):
        # assert self.shape == other.shape
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other) # 讓維度一致
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # print('self.grad = ', self.grad)
            # print('other.grad = ', other.grad)
            # print('out.grad = ', out.grad, 'op=', out._op)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other) # 讓維度一致
        # other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # print('self.shape=', self.shape)
            # print('other.shape=', other.shape)
            # print('out.shape=', out.shape)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
                        
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu') # Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def matmul(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data , other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.dot(out.grad,other.data.T)
            other.grad += np.dot(self.data.T,out.grad)            
            
        out._backward = _backward

        return out

    def softmax(self):
        out =  Tensor(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
        softmax = out.data

        def _backward():
            s = np.sum(out.grad * softmax, 1)
            t = np.reshape(s, [-1, 1]) # reshape 為 n*1
            self.grad += (out.grad - t) * softmax

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data),(self,),'log')

        def _backward():
            self.grad += out.grad/self.data
        out._backward = _backward

        return out    
    
    def sum(self,axis = None):
        out = Tensor(np.sum(self.data,axis = axis), (self,), 'SUM')
        
        def _backward():
            output_shape = np.array(self.data.shape)
            output_shape[axis] = 1
            tile_scaling = self.data.shape // output_shape
            grad = np.reshape(out.grad, output_shape)
            self.grad += np.tile(grad, tile_scaling)
            
        out._backward = _backward

        return out

    def cross_entropy(self, yb): 
        epsilon = 1e-12 
        ce = -(yb * np.log(self.data + epsilon)).sum(axis=-1).sum()  
        return ce

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            #print(v)
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\06\learn> python .\micro.py
2024-06-18 22:36:39.316420: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2024-06-18 22:36:39.316994: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
epoch 1, accuracy: 96.10%
epoch 2, accuracy: 96.94%
epoch 3, accuracy: 97.09%
epoch 4, accuracy: 96.92%
epoch 5, accuracy: 97.84%
epoch 6, accuracy: 97.51%
```
### 習題七
> 自己定義一個神經網路模型，並在 MNIST 資料集上訓練並跑出正確率

```py
# 參考 https://github.com/ccc112b/py2cs/blob/master/03-人工智慧/06-強化學習/01-強化學習/01-gym/04-run/cartpole_human_run.py
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
...
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
> 自己設計 RAG 或 ReAct 的程式(可以用 langchain 或 dspy)

```py
# 參考 https://github.com/ali1234-56/ai-homework/blob/main/hk10/rag.py
import os
import openai
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

openai.api_key = os.environ["OPENAI_API_KEY"]

prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = [TavilySearchResults(max_results=2)]

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_input = input("Enter your question: ")

response = agent_executor.invoke({"input": user_input + "，#zh-TW"})
user_input1 = response['output'] + " 將任何輸入翻譯成繁體中文"

response1 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input1}
    ]
)
print(response1['choices'][0]['message']['content'])

```
```
PS C:\Users\alex2\Desktop\NQU\ai\習題\10> python .\rag.py
C:\Users\alex2\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\langchain\chat_models\__init__.py:32: LangChainDeprecationWarning: Importing chat models from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:

`from langchain_community.chat_models import ChatOpenAI`.

To install langchain-community run `pip install -U langchain-community`.
  warnings.warn(
C:\Users\alex2\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\langchain_core\_api\deprecation.py:139: LangChainDeprecationWarning: The class `ChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 0.3.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import ChatOpenAI`.
  warn_deprecated(
Enter your question: 台灣總統?


> Entering new AgentExecutor chain...
I should use the search engine to find out who the current president of Taiwan is.
Action: tavily_search_results_json
Action Input: Current president of Taiwan[{'url': 'https://en.wikipedia.org/wiki/Tsai_Ing-wen', 'content': 'On 9 August 2020, the United States Health and Human Services Secretary Alex Azar of the Trump administration became the highest-level Cabinet member to visit Taiwan since the diplomatic break between the ROC and the United States in 1979.[93]\nIn April 2021, the United States ambassador to Palau made an official visit to Taiwan, the first time a US ambassador had done so since the US switched recognition from the ROC to the PRC in 1979.[94]\nIn the same month, the United States President Joe Biden also sent an official delegation including former senator Chris Dodd to Taiwan.[95]\nOn November 3, 2021 the first official European Union delegation arrived in Taiwan led by French MEP Raphael Glucksmann, and consisting of Lithuanian MEPs Andrius Kubilius and Petras Auštrevičius, Czech MEP Markéta Gregorová, Austrian MEP Andreas Schieder, Greek MEP Georgios Kyrtsos and Italian MEP Marco Dreosto, with the purpose of conducting exchanges on disinformation and cyber attacks against democracies.[96][97]\nThe visit followed an official tour of Central Europe by foreign minister Joseph Wu which included an unofficial visit to Brussels.[98][99]\nOn August 2, 2022, U.S. House speaker Nancy Pelosi visited Taiwan with a delegation of 6\nDemocratic representatives, the first since a visit by Newt Gingrich in 1997, and the highest-profile visit since.\n On 27 April 2011, Tsai became the first female presidential candidate in Taiwan after she defeated former Premier Su Tseng-chang by a small margin in a nationwide phone poll (of more than 15,000 samples) that served as the party\'s primary.[3] Tsai ran against incumbent President Ma Ying-jeou of the Kuomintang and James Soong of the People First Party in the 5th direct presidential election, which was held on 14 January 2012.[57] Garnering 45% of the vote, she conceded defeat to President Ma in an international press conference, resigning her seat as Chairman of the DPP.[58]\n2016[edit]\nOn 15 February 2015, Tsai officially registered for the Democratic Progressive Party\'s presidential nomination primary.[59] Though Lai Ching-te and Su Tseng-chang were seen as likely opponents,[60] Tsai was the only candidate to run in the primary and the DPP officially nominated her as the presidential candidate on 15 April.[61][62]\n The Central Epidemic Command Center was activated on January 20, 2020, and deactivated May 1, 2023.[106][107]\nTrade relations[edit]\nOn August 28, 2020, the Tsai administration lifted a ban on leaning agent ractopamine, clearing the way for U.S. pork imports and removing a major hurdle for bilateral trade talks between Taiwan and the United States.[108]\nThis move proved controversial domestically, and a referendum to reinstate the ban was defeated in 2021.[109]\nOn June 1, 2022, Taiwan established a trade negotiation framework titled the U.S.-Taiwan Initiative on 21st-Century Trade.[110]\nOn June 1, 2023, an initial trade agreement was signed with the United States on June 1, 2023 under this framework, which streamlined customs regulations, established common regulatory practices, and introduced anti-corruption measures,[111][112] with further measures still in discussion.[113]\nEnergy policy[edit]\nThe Tsai administration has stated an electricity supply goal of 20% from renewables, 30% from coal and 50% from liquefied natural gas by 2025.[114]\nBills under the umbrella of the Forward-Looking Infrastructure initiative have been used to fund green energy initiatives.\n The Act Governing the Handling of Ill-gotten Properties by Political Parties and Their Affiliate Organizations was passed in July and Wellington Koo, one of the main authors of the Act, was named as the committee chairman in August.[183][184]\nThe stated goal of the act is to investigate state assets which were illegally transferred to private political parties and affiliates during the martial law era, and therefore applies only to political parties officially formed before the end of martial law.[185]\nThis effectively limits its scope to the KMT, which has insisted that it has been illegally and unconstitutionally persecuted and that the investigation is a political witch hunt.[186][187] She generally supports the diversification of Taiwan\'s economic partners.[204][205]\nIn response to the death of Chinese Nobel Peace Prize laureate Liu Xiaobo, who died of organ failure while in government custody, Tsai pleaded with the Communist government to "show confidence in engaging in political reform so that the Chinese can enjoy the God-given rights of freedom and democracy".[206]\nTsai has accused the Communist Party of China\'s troll army of spreading fake news via social media to influence voters and support candidates more sympathetic to Beijing ahead of the 2018 Taiwanese local elections.[207][208][209]\nIn January 2019, Xi Jinping, General Secretary of the Chinese Communist Party (CCP), had announced an open letter to Taiwan proposing a one country, two systems formula for eventual unification.'}, {'url': 'https://apnews.com/article/taiwan-president-lai-chingte-af4ff1f254e11c24f751cea000fb42d9', 'content': "10 of 19 |. In this photo released by the Taipei News Photographer, President-elect Lai Ching-te, also known by his English name William, gets sworn in as Taiwan's new president during his inauguration ceremony in Taipei, Taiwan, Monday, May 20, 2024. Lai was sworn in as Taiwan's new president Monday, beginning a term in which he is ..."}]I now know the final answer
Final Answer: The current president of Taiwan is Lai Ching-te.

> Finished chain.
台灣目前的總統是賴清德。
```
### 期中
* [人工智慧的筆記](https://hackmd.io/@Jung217/nqu_ai)
* [基於 LSTM & BERT 機器學習之網路輿情分析](https://github.com/Jung217/LSTM_BERT_Sentiment_Analysis)
* [Jung217 / openai_agent_linebot](https://github.com/Jung217/openai_agent_linebot)