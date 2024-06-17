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