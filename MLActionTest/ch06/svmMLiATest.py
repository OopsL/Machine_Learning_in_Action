from ch06 import svmMLiA

dataMat,labelMat = svmMLiA.loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch06/testSet.txt')
# print(labelMat)
# b,alphas = svmMLiA.smoSimple(dataMat,labelMat,0.6,0.001,40)
# print('b = ',b)
# print(alphas[alphas>0])
# for i in range(100):
#     if alphas[i] > 0:
#         print(dataMat[i],labelMat[i])

# b,alphas = svmMLiA.smoP(dataMat,labelMat,0.6,0.001,40)
# print('b = ',b)
# print(alphas[alphas>0])
# for i in range(100):
#     if alphas[i] > 0:
#         print(dataMat[i],labelMat[i])


svmMLiA.testRbf(0.1)