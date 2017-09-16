from ch02 import KNN

# group,labels = KNN.createDataset()
# result = KNN.classify0([1,1],group,labels,3)
# print(result)

print('---------约会数据加载---------')

datingDataMat,datingLabels = KNN.file2matrix('/Users/JD.K/Downloads/machinelearninginaction/Ch02/datingTestSet2.txt')
# print(datingDataMat)
# print(datingLabels)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0 * np.array(datingLabels),15.0 * np.array(datingLabels))
# plt.show()

# print('-----------归一化-------')
# normMat,ranges,minVals = KNN.autoNorm(datingDataMat)
# print(normMat)

# print('-----------测试-------')
# errorRatio = KNN.datingClassTest()
# print(errorRatio)

# print('-----------预测-------')
# KNN.classififyPerson()

# print('-----------数字识别-------')
# KNN.handwritingClassTest()
