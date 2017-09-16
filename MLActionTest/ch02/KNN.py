import numpy as np
import operator as op

_filePath = '/Users/JD.K/Downloads/machinelearninginaction/Ch02/datingTestSet2.txt'

def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    # 将inX切为4行,与dataset对应,并求差
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet

    # 同一行数据分别平方
    sqDiffMat = diffMat **2
    # 同一行数据平方后求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号
    distances = sqDistances ** 0.5
    sortedDisIndicies = np.argsort(distances)
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDisIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=lambda d:d[1],reverse = True)
    print(sortedClassCount)
    print(type(sortedClassCount))
    return sortedClassCount[0][0]

# 将输入的字符串转换为矩阵或类标签向量
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numOfLines = len(arrayOfLines)
    returnMat = np.zeros((numOfLines,3))
    classLabelsVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelsVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelsVector

# 归一化数值
def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    m = dataset.shape[0]
    normDataset = np.zeros(np.shape(dataset))
    normDataset = dataset - np.tile(minVals,(m,1))
    normDataset = normDataset / np.tile(ranges,(m,1))
    return normDataset,ranges,minVals

# KNN分类测试代码
def datingClassTest():
    hoRaito = 0.05
    datingDataMat,datingLabels = file2matrix('/Users/JD.K/Downloads/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = np.int16(m * hoRaito)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print("the classifier came back with %d,the real answer is: %d" % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1
    print('the total error rate is : %f'% (errorCount/float(numTestVecs)))


def classififyPerson():
    resultList = ['not at all','in small doess','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix(_filePath)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([percentTats,ffMiles,iceCream])
    classfierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person: " + resultList[classfierResult -1])



def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


import os
_digitPath = '/Users/JD.K/Downloads/machinelearninginaction/Ch02/digits/trainingDigits'
_testDigitPath = '/Users/JD.K/Downloads/machinelearninginaction/Ch02/digits/testDigits'
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir(_digitPath)
    m = len(trainingFileList)
    trainingSet = np.zeros((m,1024))
    #     遍历所有文件名,获取对应的数字
    for i in  range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr= int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingSet[i,:] = img2vector(_digitPath +'/'+fileNameStr)
    testFileList = os.listdir(_testDigitPath)
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileName = fileNameStr.split('.')[0]
        classNumStr = int(fileName.split('_')[0])
        vectorUnderTest = img2vector(_testDigitPath+'/'+fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingSet,hwLabels,3)
        print("the classifier came ack with : %d,the real answer is : %d" % (classifierResult,classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1
    print("\nthe total number of errors is: %d" %errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))












