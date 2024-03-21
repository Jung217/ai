import random as rd
citys = [
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
]
Ccity = [
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
]

path = [i for i in range(len(citys))]
print(path)

def distance(p1, p2):
    #print('p1=', p1)
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def pathLength(p):
    dist = 0
    plen = len(p)
    for i in range(plen):
        dist += distance(citys[p[i]], citys[p[(i+1)%plen]])
    return dist

def CpathLength(p):
    dist = 0
    plen = len(p)
    for i in range(plen):
        dist += distance(Ccity[p[i]], Ccity[p[(i+1)%plen]])
    return dist

def ex(arr, max=20000):
    cont = 0
    while(cont < max):
        clen = Clen = 0.0
        Ccity = arr.copy()
        arrlen = len(Ccity)-1
        RD = rd.randint(0, arrlen)
        if RD == arrlen :
            tmp = Ccity[0] 
            Ccity[0] = Ccity[RD]
            Ccity[RD] = tmp
        else :
            tmp = Ccity[RD] 
            Ccity[RD] = Ccity[RD+1]
            Ccity[RD+1] = tmp
        path = [i for i in range(len(arr))]
        Cpath = [i for i in range(len(Ccity))]
        clen, Clen= pathLength(path), CpathLength(Cpath)
        print(Clen)
        if Clen < clen : arr = Ccity # clen == Clen ???
        else : cont += 1
    #Rpath = [i for i in range(len(Ccity))]
    return Ccity

print(ex(citys))

