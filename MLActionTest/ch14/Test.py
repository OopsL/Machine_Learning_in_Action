from ch14 import svdRec
import numpy as np

myMat = np.mat(svdRec.loadExData())
myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
myMat[3,3] = 2

print(myMat)

rec = svdRec.recommend(myMat,1,estMethod=svdRec.svdEst)
print(rec)
















