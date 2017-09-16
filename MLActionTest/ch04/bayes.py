import numpy as np


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# def setOfWord2Vec(vocabList,inputSet):
#     returnVec = [0]*len(vocabList)
#     for word in inputSet:
#         if word in vocabList:
#             returnVec[vocabList.index(word)] = 1
#         else: print("eht word %s is not in mu Vocabulary" % word)
#
#     return returnVec

def setOfWord2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else: print("eht word %s is not in mu Vocabulary" % word)

    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive

def classify(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    if p1 > p0: return 1
    else: return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList,testEntry))
    print(testEntry ,'classified as:',classify(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList,testEntry))
    print(testEntry ,'classified as:', classify(thisDoc, p0V, p1V, pAb))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = [];classList = []; fullTest = []
    for i in range(1,26):
        wordList = textParse(open('/Users/JD.K/Downloads/machinelearninginaction/Ch04/email/spam/%d.txt' % i,encoding = "ISO-8859-1").read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList = textParse(open('/Users/JD.K/Downloads/machinelearninginaction/Ch04/email/ham/%d.txt' % i,encoding = "ISO-8859-1").read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(0,50);testSet = []
    for i in range(10):
        randomIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randomIndex])
        del(list(trainingSet)[randomIndex])
    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for dcoIndex in testSet:
        wordVector = setOfWord2Vec(vocabList,docList[docIndex])
        if classify(np.array(wordVector),p0V,p1V,pAb) != classList[docIndex]:
            errorCount += 1
    print('the error rate is : ',float(errorCount)/len(testSet))








