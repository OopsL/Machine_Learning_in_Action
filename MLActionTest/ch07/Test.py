from ch07 import adaboost
import numpy as np

dataMat = np.matrix([[1.0,2.1],[2,1.1],[1.3,1],[1,1],[2,1]])
classLabels = [1.0,1.0,-1.0,-1.0,1.0]
# print(dataMat)
# print(dataMat[:,0] <=1.5)
# D = np.mat(np.ones((5,1))/5.0)
# print(adaboost.buildStump(dataMat,classLabels,D))

# classifierArray = adaboost.adaBoostTrainDS(dataMat,classLabels,300)
# print(adaboost.adaClassify([[5,5],[0,0]],classifierArray))

dataArr,labelArr = adaboost.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch07/horseColicTraining2.txt')
classifierArray = adaboost.adaBoostTrainDS(dataArr,labelArr,50)

dataTest,labelTest = adaboost.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch07/horseColicTest2.txt')
predict = adaboost.adaClassify(dataTest,classifierArray)

errArr = np.mat(np.ones((67,1)))
print(errArr[predict != np.mat(labelTest).T].sum()/67)




