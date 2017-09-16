from ch03 import trees
from ch03 import treePlotter

myDat,labels = trees.createDataSet()
# print(myDat)
# myDat[0][-1] = 'maybe'
# shannonEnt = trees.calShannonEnt(myDat)
# print(shannonEnt)
# bestFeature = trees.chooseBestFeatureToSplit(myDat)
# print("最好的特征是: %d" % bestFeature)
#
# # a = ['no','yes']
# # del 'no'
# # print(a)
#
# b = [[1,2,3]]
# print(len(b[0]))

myTree = treePlotter.retriveTree(0)
print(type(myTree))
# print(myTree)
# classLabel = trees.classify(myTree,labels,[1,1])
# print(classLabel)

# trees.storeTree(myTree,"/Users/JD.K/Desktop/classifierStorage.txt")
# loadTree = trees.grabTree("/Users/JD.K/Desktop/classifierStorage.txt")
# print(loadTree)

fr = open('/Users/JD.K/Downloads/machinelearninginaction/Ch03/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescrip','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)