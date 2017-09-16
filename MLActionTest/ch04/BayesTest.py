from ch04 import bayes
import numpy as np

# listrOPosts,listClasses = bayes.loadDataSet()
# myVocList = bayes.createVocabList(listrOPosts)
# print(myVocList)
# trainMat = []
# for postinDoc in listrOPosts:
#     trainMat.append(bayes.setOfWord2Vec(myVocList,postinDoc))
#
# p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)
#
# print(p0V)
# print(p1V)

# bayes.testingNB()
bayes.spamTest()