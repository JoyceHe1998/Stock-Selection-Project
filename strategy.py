# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
from mathCal import *
from sklearn.cluster import KMeans


class strategy:
    def __init__(self, Cl, Op, Vol, mDate, stks, ListDate, nTrain=20, nStock=5, nCluster=5):
        self.Cl = Cl                
        self.Op = Op                
        self.Vol = Vol              
        self.mDate = mDate          
        self.stks = stks         
        self.ListDate = ListDate    
        self.T = len(mDate)        
        self.nTrain = nTrain       
        self.nStock = nStock        
        self.nCluster = nCluster    

        self.shortROC = calROC(Cl, period=20)
        self.longROC = calROC(Cl, period=125)
        self.shortTr = calTrend(Cl, period=50)
        self.longTr = calTrend(Cl, period=200)

    def stockPool(self, t):
        curDate = self.mDate.iloc[t,:]
        stkPool = []

        for stk in range(0,len(self.stks)):
            stkPool.append(self.stks[0][stk])

                       
        self.stkPool = stkPool
        type(self.stkPool) 

    def calReturn(self, stk, t):
        return self.Cl[stk].iat[t] / self.Op[stk].iat[t - self.nTrain + 1] - 1

    def getFeatures(self, t, stkPool):
        return np.array([self.shortROC[stkPool].iloc[t],
                         self.longROC[stkPool].iloc[t],
                         self.shortTr[stkPool].iloc[t],
                         self.longTr[stkPool].iloc[t]]).transpose()

    def calEuroDistance(self, Features, center):
        return np.apply_along_axis(lambda f: np.linalg.norm(f - center), 1, Features)

    def trainModel(self, t):
        Features = self.getFeatures(t - self.nTrain, self.stkPool)
        kmeans = KMeans(n_clusters=self.nCluster, random_state=0).fit(Features)
        
        classes = kmeans.labels_              
        centers = kmeans.cluster_centers_     

        numOfClasses = np.zeros(self.nCluster)  
        resOfClasses = np.zeros(self.nCluster)  

        for i in range(len(classes)):
            stk = self.stkPool[i]
            cls = classes[i]
            numOfClasses[cls] += 1
            resOfClasses[cls] += self.calReturn(stk, t)

        avgResOfClasses = resOfClasses / numOfClasses       
        bestClass = np.argmax(avgResOfClasses)            
        bestCenter = centers[bestClass]                     

        newFeatures = self.getFeatures(t, self.stkPool)    
        dists = self.calEuroDistance(newFeatures, bestCenter)
        index = np.argsort(dists)[:self.nStock]            

        return index

    def trade(self):
        tradeLog = pd.DataFrame(np.zeros(self.Cl.shape), index=self.Cl.index, columns=self.Cl.columns)

        for t in range(self.nTrain, self.T):

            if t % 20 == 0:
                print(t)
            self.stockPool(t)
            index = self.trainModel(t)
            stocks = set(np.array(self.stkPool)[index])    
            tradeLog.loc[self.Cl.index[t], stocks] = 1

        self.tradeLog = tradeLog

    def run(self):
        self.trade()
        return self.tradeLog

if __name__ == '__main__':
    s = strategy(Cl, Op, Vol, mDate, stks, ListDate, nCluster=10)
    tradeLog = s.run()
