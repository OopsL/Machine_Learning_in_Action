import numpy as np
from numpy import linalg as la

def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

# 范数计算
def euclidSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA,inB))

# 皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA,inB,rowvar=0)[0][1]


# 余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


def standEst(dataMat,user,simMeas,item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
        if simTotal ==0: return 0
        else: return ratSimTotal/simTotal


def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0: print("you rated everything")
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

def svdEst(dataMat,user,simMeas,item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    sig4 = np.mat(np.eye(4)*Sigma[:4])
    xformedItems = dataMat.T*U[:,:4]*sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating==0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print("the %d and %d similarity is: %f" % (item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
        if simTotal == 0: return 0
        else: return ratSimTotal/simTotal











