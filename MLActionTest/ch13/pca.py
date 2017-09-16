import numpy as np


def loadDataSet(filename,delim='\t'):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        lineArr = []
        for i in range(len(curline)):
            lineArr.append(float(curline[i]))
        # flt = map(float,curline)
        dataMat.append(lineArr)
    return np.mat(dataMat)

def pca(dataMat,topNfeat=100):
    meanVals = np.mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    print('topNfeat: ',-(topNfeat+1))
    redEigVects = eigVects[:,eigValInd]
    lowDataMat = meanRemoved * redEigVects
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat, reconMat


def replaceNanWithMean():
    dataMat = loadDataSet('/Users/JD.K/Downloads/machinelearninginaction/Ch13/secom.data')
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:,i].A)),i])
        dataMat[np.nonzero(np.isnan(dataMat[:,i].A))[0],i] = meanVal
    return dataMat


