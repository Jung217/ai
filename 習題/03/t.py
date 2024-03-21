from random import choice

def f(xyz:list[float])->float:
    return 3*xyz[0]+2*xyz[1]+5*xyz[2]

def limit(xyz:list[float])->bool:
    for i in xyz:
        if i < 0:
            return False
    if xyz[0]+xyz[1]>10 or 2*xyz[0]+xyz[2]>9 or xyz[1]+2*xyz[2]>11:
        return False
    return True

STEP=0.01
# 爬山演算法：找出可能最佳解
def hillClimbing(equation:list[float])->list[float]:
    fail=0
    while fail<100000:
        #print(equation)
        temp=equation.copy()
        for i in range(len(temp)):
            temp[i]+=STEP*choice([1,-1])
        if f(temp)>f(equation) and limit(temp):
            equation=temp
            fail=0
            print(equation,f(equation))
        else:
            fail+=1
    return equation

ans=hillClimbing([2,2,2])
print(ans,f(ans))