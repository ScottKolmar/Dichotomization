import numpy as np
import pandas as pd
from random import sample
from scipy.stats.stats import pearsonr, ttest_ind

class multidataPopulation():
    """

    """
    def __init__(self, size, mean, cov):
        nums = np.random.multivariate_normal(mean = mean, cov = cov, size = size).T
        if len(nums) > 2:
            numdict = {}
            for i, num in enumerate(nums):
                if i == len(nums) - 1:
                    numdict['y'] = num
                else:
                    numdict['x{}'.format(i)] = num

        for key in numdict.keys():
            self.key = numdict[key]

class threeVarDataPopulation():
    """

    """

    def __init__(self, size, mean, cov):
        x1, x2, y = np.random.multivariate_normal(mean=mean, cov=cov, size=size).T
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.x1x2y = list(zip(self.x1, self.x2, self.y))

class dataPopulation():
    """

    """
    def __init__(self, size, mean, cov):
        x1,y = np.random.multivariate_normal(mean = mean, cov = cov, size = size).T
        self.x1 = x1
        self.y = y
        self.x1y = list(zip(self.x1, self.y))

class samplePopulation():
    """

    """
    def __init__(self, n, dataPopulation):
        self.x1y = sample(dataPopulation.x1y, n)
        self.x1 = list(zip(*self.x1y))[0]
        self.y = list(zip(*self.x1y))[-1]
        self.corrx1y, self.corrx1yp = pearsonr(self.x1, self.y)
        self.x1ydich = None
        self.ydich = None
        self.corrx1ydich = None
        self.corrx1ydichp = None
        self.t = None
        self.tp  = None

    def getDichStats(self, perc):
        ydich = np.digitize(self.y, bins = [np.percentile(self.y, perc)])
        self.ydich = ydich
        self.corrx1ydich, self.corrx1ydichp = pearsonr(self.x1, self.ydich)
        self.x1ydich = list(zip(self.x1, self.ydich))
        self.y0 = [x[0] for x in self.x1ydich if x[1] == 0]
        self.y1 = [x[0] for x in self.x1ydich if x[1] == 1]
        self.t, self.tp = ttest_ind(self.y0, self.y1, equal_var= False)

class threeVarSamplePopulation():
    def __init__(self, n, threeVarDataPopulation):
        self.x1x2y = sample(threeVarDataPopulation.x1x2y, n)
        self.x1 = list(zip(*self.x1x2y))[0]
        self.x2 = list(zip(*self.x1x2y))[1]
        self.y = list(zip(*self.x1x2y))[-1]
        self.corrx1y, self.corrx1yp = pearsonr(self.x1, self.y)
        self.corrx1x2, self.corrx1x2p = pearsonr(self.x1, self.x2)
        self.corrx2y, self.corrx2yp = pearsonr(self.x2, self.y)
        self.x1x2ydich = None
        self.ydich = None
        self.corrx1ydich = None
        self.corrx1ydichp = None
        self.corrx2ydich = None
        self.corrx2ydichp = None
        self.t = None
        self.tp  = None

    def getDichStats(self, perc):
       ydich = np.digitize(self.y, bins=[np.percentile(self.y, perc)])
       self.ydich = ydich
       self.corrx1ydich, self.corrx1ydichp = pearsonr(self.x1, self.ydich)
       self.x1x2ydich = list(zip(self.x1, self.x2, self.ydich))
       self.y0 = [x[0] for x in self.x1x2ydich if x[-1] == 0]
       self.y1 = [x[0] for x in self.x1x2ydich if x[-1] == 1]
       self.t, self.tp = ttest_ind(self.y0, self.y1, equal_var=False)
