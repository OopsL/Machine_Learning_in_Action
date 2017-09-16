from ch13 import pca
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


dataMat = pca.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch13/testSet.txt')
lowMat,reconMat = pca.pca(dataMat,2)
print(np.shape(lowMat))

a = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
print(a[:-4:-1])

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
# ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
# plt.show()