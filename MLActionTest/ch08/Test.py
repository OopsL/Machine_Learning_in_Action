from ch08 import regression
import numpy as np
import matplotlib.pyplot as plt


# xArr,yArr = regression.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch08/ex0.txt')
# # ws = regression.standRegres(xArr,yArr)
# # xMat = np.mat(xArr)
# # yMat = np.mat(yArr)
# # yHat = xMat*ws
#
# yHat = regression.lwlrTest(xArr,xArr,yArr,0.01)
# xMat = np.mat(xArr)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# print(xMat[srtInd])
# print(xSort)
#
#
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# # xCopy = xMat.copy()
# # xCopy.sort(0)
# # yHat = xCopy*ws
# # ax.plot(xCopy[:,1],yHat)
# # plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c='red')
#
# plt.show()

abX,abY = regression.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch08/abalone.txt')
ridgeWeight = regression.ridgeTest(abX,abY)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeight)
plt.show()











