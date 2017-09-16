import numpy as np


def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        dataMat.append([float(curline[0]),float(curline[1])])
        labelMat.append(float(curline[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j = i
    while j == i:
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L ==H: print('L == H'); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print('eta >= 0'); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j] - alphaJold) < 0.00001:print('j not moving enough');continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print('iter: %d i %d,pairs changed %d' % (iter,i,alphaPairsChanged))
        if alphaPairsChanged == 0: iter += 1
        else: iter = 0
        print('iteration number : %d' % iter)
    return b,alphas


# #############  完整版的 Platt SMO  ########################

# 支持函数
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X= dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

# def calcEk(oS,k):
#     fXk = float(np.multiply(oS.alphas,oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
#     Ek = fXk - float(oS.labelMat[k])
#     return Ek

# 核函数版
def calcEk(oS,k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    # print(np.nonzero(oS.eCache[:,0].A))
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

def updataEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

#     算法优化
# def innerL(i,oS):
#     Ei = calcEk(oS,i)
#     if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
#         j,Ej = selectJ(i,oS,Ei)
#         alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
#         if oS.labelMat[i] != oS.labelMat[j]:
#             L = max(0,oS.alphas[j] - oS.alphas[i])
#             H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
#         else:
#             L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
#             H = min(oS.C,oS.alphas[j] + oS.alphas[i])
#         if L == H: print('L == H'); return 0
#         eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
#         if eta >= 0: print('eta >= 0'); return 0
#         oS.alphas[j] -= oS.labelMat[j] * (Ei-Ej)/eta
#         oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
#         updataEk(oS,j)
#         if abs(oS.alphas[j] - alphaJold) < 0.00001:
#             print('j not moving enough'); return 0
#         oS.alphas[i] += oS.labelMat[j]*oS.alphas[i]*(alphaJold-oS.alphas[j])
#         updataEk(oS,i)
#         b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
#         b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
#         if 0 < oS.alphas[i] and oS.C > oS.alphas[i]: oS.b = b1
#         elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]: oS.b = b2
#         else:oS.b = (b1+b2)/2.0
#         return 1
#     else: return 0

# 核函数版
def innerL(i,oS):
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
        if L == H: print('L == H'); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0: print('eta >= 0'); return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updataEk(oS,j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enough'); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.alphas[i]*(alphaJold-oS.alphas[j])
        updataEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]: oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]: oS.b = b2
        else:oS.b = (b1+b2)/2.0
        return 1
    else: return 0

# 完整的Platter SMO外循环
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin,0')):
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True; alphaParisChanged = 0
    while iter < maxIter and (alphaParisChanged>0 or entireSet):
        alphaParisChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaParisChanged += innerL(i,oS)
                print('fullSet,iter: %d i %d, pairs changed %d' % (iter,i,alphaParisChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaParisChanged += innerL(i,oS)
                print('non-bound,iter: %d i %d, pairs changed %d' % (iter,i,alphaParisChanged))
            iter += 1
        if entireSet: entireSet = False
        elif alphaParisChanged == 0: entireSet = True
        print('iteration number: %d' % iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

# 核函数
def kernelTrans(X,A,kTup):
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else: raise NameError('Have a problem')
    return K

# 测试函数
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch06/testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200,0.0001,10000,('rbf',k1))
    dataMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    print(np.nonzero(alphas.A>0))
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % np.shape(sVs)[0])
    m,n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print('the training error rate is %f' % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch06/testSetRBF2.txt')
    errorCount = 0
    dataMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
    print("the test error rate is %f :" % (float(errorCount)/m))







