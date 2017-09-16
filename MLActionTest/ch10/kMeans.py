import numpy as np


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltLine = list(map(float,curline))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    # print(np.sum(np.power(vecA - vecB,2)))
    return np.sqrt(np.sum(np.power(vecA - vecB,2)))

def randCent(dataSet,k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI< minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment.A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def biKmeans(dataSet,k,distMeans=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,:] = distMeans(np.mat(centroid0),dataSet[j,:])**2
        while len(centList) < k:
            lowestSSE = np.inf
            for i in range(len(centList)):
                ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0],:]
                centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeans)
                sseSplit = np.sum(splitClustAss[:,1])
                sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1])
                if (sseSplit+sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit+sseNotSplit
            bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
            print("the bestCentToSplit is: ",bestCentToSplit)
            print("the len of bestClustAss is: ",len(bestClustAss))
            centList[bestCentToSplit] = bestNewCents[0,:]
            centList.append(bestNewCents[1,:])
            clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
        return centList, clusterAssment























