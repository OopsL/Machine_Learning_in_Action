import numpy as np


def loadDataSet():
    return [[1,2,3],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1= []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in transaction:
                C1.append(item)
    C1.sort()
    return list(set(C1))
    # return map(frozenset,C1)

def scanD(D,ck,minSupport):
    ssCnt = {}
    for tid in D:
        for can in ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItens = len(D)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItens
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData