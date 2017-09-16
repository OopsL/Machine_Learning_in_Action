from ch09 import regTree
import numpy as np
from Tkinter import *

# testMat = np.mat(np.eye(4))
# mat0,mat1 = regTree.binSplitDataSet(testMat,1,0.5)
# dataSet[np.nonzero(dataSet[:,feature] > value)[0],:][0]
# print(testMat[:,1] > 0.5)
# print(np.nonzero(testMat[:,1] > 0.5))
# print(testMat[np.nonzero(testMat[:,] <= 0.5)[0],:])
# print('-----')
# print(testMat[np.nonzero(testMat[:,] <= 0.5)[0],:][0])
# print(mat1)


# myDat = regTree.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch09/ex00.txt')
# myDat = np.mat(myDat)
# tree = regTree.createTree(myDat)
# print(tree)



# myDat2 = regTree.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch09/ex2.txt')
# myDat2 = np.mat(myDat2)
# tree = regTree.createTree(myDat2,ops=(0,1))
#
# print(tree)
# print('-----------')
#
# myTestData = regTree.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch09/ex2test.txt')
# myTestData = np.mat(myTestData)
#
# treeMerge = regTree.prune(tree,myTestData)
# print(treeMerge)


# myMat2 = np.mat(regTree.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch09/exp2.txt'))
# tree = regTree.createTree(myMat2,regTree.modelLeaf,regTree.modelErr,(1,10))
# print(tree)


trainMat = np.mat(regTree.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch09/bikeSpeedVsIq_train.txt'))
testMat = np.mat(regTree.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch09/bikeSpeedVsIq_test.txt'))
myTree = regTree.createTree(trainMat,ops=(1,20))
yHat = regTree.createForeCast(myTree,testMat[:,0])
print(np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])


myTree = regTree.createTree(trainMat,regTree.modelLeaf,regTree.modelErr,(1,20))
yHat = regTree.createForeCast(myTree,testMat[:,0],regTree.modelTreeEval)
print(np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1])


















