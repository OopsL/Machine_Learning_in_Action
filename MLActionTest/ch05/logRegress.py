import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('/Users/JD.K/Downloads/machinelearninginaction/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# 画出拟合直线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s') #画离散的点
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    weights = np.array(weights)
    y = (-weights[0] -weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# def stocGradAscent0(dataMatrix,classLabels):
#     m,n = np.shape(dataMatrix)
#     alpha = 0.01
#     weights = np.ones(n)
#     for i in range(m):
#         h = sigmoid(sum(dataMatrix[i] * weights))
#         error = classLabels[i] - h
#         weights = weights + alpha * error * dataMatrix[i]
#     return weights

def stocGradAscent1(dataMatrix,classLabels,numItem=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numItem):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randomIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randomIndex] * weights))
            error = classLabels[randomIndex] - h
            weights = weights + alpha * error * dataMatrix[randomIndex]
            del(list(dataIndex)[randomIndex])
    return weights

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def coliTest():
    frTrain = open('/Users/JD.K/Downloads/machinelearninginaction/Ch05/horseColicTraining.txt')
    frTest = open('/Users/JD.K/Downloads/machinelearninginaction/Ch05/horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split('\t')
        # print('curline type : ',type(curLine))
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    # print(trainingSet)
    # print(np.array(trainingSet))
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(curLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the errorrate is %f' % errorRate)
    return errorRate

def multiTest():
    numTest = 10; errorSum = 0.0
    for k in range(numTest):
        errorSum += coliTest()
    print('after %d iterations the average error rate is : %f' % (numTest,errorSum/float(numTest)))











