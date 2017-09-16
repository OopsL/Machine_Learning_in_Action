import numpy as np

# def loadDataSet(filename):
#     dataMat = []
#     fr = open(filename)
#     for line in fr.readlines():
#         curline = line.strip().split('\t')
#         flt = map(float,curline)
#         dataMat.append(flt)
#     return dataMat

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline)):
            lineArr.append(float(curline[i]))
        # flt = map(float,curline)
        dataMat.append(lineArr)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def  regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

def chooseBestSplit(dataSet,learType=regLeaf,errType=regErr,ops=(1,4)):
    tols = ops[0]; tolN = ops[1]

    trans = dataSet[:,-1].T.tolist()[0]

    if len(set(trans)) == 1:
        return None,learType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    for i in range(n-1):
        for splitValue in set(dataSet[:,i].T.tolist()[0]):
            mat0,mat1 = binSplitDataSet(dataSet,i,splitValue)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = i
                bestValue = splitValue
    if (S - bestS) < tols:
        return None,learType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None,learType(dataSet)
    return bestIndex,bestValue

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree,testData):
    if np.shape(testData)[0] == 0: return getMean(tree)
    if isTree(tree['left']) or isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errNoMerge = sum(np.power(lSet[:,-1]-tree['left'],2)) + sum(np.power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errMerge = sum(np.power(testData[:,-1] - treeMean,2))
        if errMerge < errNoMerge:
            print('merging')
            return treeMean
        else: return tree
    else: return tree

def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        print("this matris is singula, cannot do inverse, try increasing the second value of ops")
    ws = xTx.I*(X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet);
    yHat = X * ws
    return sum(np.power(Y - yHat,2))

# 预测代码
def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)


def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat



















