from ch10 import kMeans
import numpy as np

# dataMat = np.mat(kMeans.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch10/testSet.txt'))
# print(kMeans.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch10/testSet.txt'))
# print(min(dataMat[:,0]))
# print(type(min(dataMat[:,0])))
# print(kMeans.randCent(dataMat,2))
# print(kMeans.distEclud(dataMat[0],dataMat[1]))



# centroids,clusterAss = kMeans.kMeans(dataMat,4)


dataMat3 = np.mat(kMeans.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch10/testSet2.txt'))
centList,myNewAssments = kMeans.biKmeans(dataMat3,3)
print(centList)


