from ch11 import apriori
import numpy as np


dataSet = apriori.loadDataSet()
C1 = apriori.createC1(dataSet)
print(C1)
