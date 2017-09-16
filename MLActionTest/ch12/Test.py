from ch12 import fpGrowth
import numpy as np

# rootNode = fpGrowth.treeNode('pyramid',9,None)
# rootNode.children['eye'] = fpGrowth.treeNode('eye',13,None)
# rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix',3,None)
# childEye = rootNode.children['eye']
# childEye.children['last'] = fpGrowth.treeNode('last',5,None)
#
# rootNode.disp()

simpDat = fpGrowth.loadSimpDat()
# fpGrowth.createTree(simpDat)
initSet = fpGrowth.createInitSet(simpDat)
# print(initSet)
myFPtree,myHeaderTab = fpGrowth.createTree(initSet,3)
# print(myFPtree.disp())
#
# conPats = fpGrowth.findPrefixPath('r',myHeaderTab['r'][1])
# print(conPats)

freqItems = []
fpGrowth.mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)

