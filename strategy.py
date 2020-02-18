# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime
from mathCal import *
from sklearn.cluster import KMeans


class strategy:
    def __init__(self, Cl, Op, Vol, mDate, stks, ListDate, nTrain=20, nStock=5, nCluster=5):
        self.Cl = Cl                # 收盘价
        self.Op = Op                # 开盘价
        self.Vol = Vol              # 成交量
        self.mDate = mDate          # 交易日期列表
        self.stks = stks            # 股票代码列表
        self.ListDate = ListDate    # 上市日期
        self.T = len(mDate)         # 总交易时间

        self.nTrain = nTrain        # 回望周期长度
        self.nStock = nStock        # 每期交易的股票数
        self.nCluster = nCluster    # 聚类个数

        self.shortROC = calROC(Cl, period=20)
        self.longROC = calROC(Cl, period=125)
        self.shortTr = calTrend(Cl, period=50)
        self.longTr = calTrend(Cl, period=200)

    def stockPool(self, t):
        '''
        t时刻的待选股票池
        '''
        curDate = self.mDate.iloc[t,:]
        stkPool = []

        for stk in range(0,len(self.stks)):
#            print(list(self.stks))
#            print(curDate)
#            print(curDate - self.ListDate[stk])
            
#            if curDate - self.ListDate[stk] < datetime.timedelta(days=365):
#                continue
#            rangeVol = self.Vol[stk].iloc[(t - self.nTrain + 1):(t + 1)]    # [t:(t-self.nTrain):-1]
#            if rangeVol.min() == rangeVol.max():    # == 0, 停牌
#                continue
            stkPool.append(self.stks[0][stk])

                       
        self.stkPool = stkPool
        type(self.stkPool) 

    def calReturn(self, stk, t):
        '''
        计算股票训练时间的收益
        '''
        return self.Cl[stk].iat[t] / self.Op[stk].iat[t - self.nTrain + 1] - 1

    def getFeatures(self, t, stkPool):
        '''
        得到t时刻的特征
        '''
        return np.array([self.shortROC[stkPool].iloc[t],
                         self.longROC[stkPool].iloc[t],
                         self.shortTr[stkPool].iloc[t],
                         self.longTr[stkPool].iloc[t]]).transpose()

    def calEuroDistance(self, Features, center):
        '''
        计算欧式距离
        '''
        return np.apply_along_axis(lambda f: np.linalg.norm(f - center), 1, Features)

    def trainModel(self, t):
        '''
        训练模型
        '''
        Features = self.getFeatures(t - self.nTrain, self.stkPool)


        # TODO: 起始点
        kmeans = KMeans(n_clusters=self.nCluster, random_state=0).fit(Features)
        
        classes = kmeans.labels_                # 分类标签
#        print(classes)
        centers = kmeans.cluster_centers_       # 分类中心


        numOfClasses = np.zeros(self.nCluster)  # 记录每个类别下的股票个数
        resOfClasses = np.zeros(self.nCluster)  # 记录每个类别下的股票收益

        for i in range(len(classes)):
            stk = self.stkPool[i]
            cls = classes[i]
            numOfClasses[cls] += 1
            resOfClasses[cls] += self.calReturn(stk, t)

        avgResOfClasses = resOfClasses / numOfClasses       # 计算每个投资组合的收益
        bestClass = np.argmax(avgResOfClasses)              # 找到最优的类
        bestCenter = centers[bestClass]                     # 最优的类的中心

        newFeatures = self.getFeatures(t, self.stkPool)     # t时刻当前的特征集
        dists = self.calEuroDistance(newFeatures, bestCenter)
        index = np.argsort(dists)[:self.nStock]             # t时刻与bestCenter最接近的nStock个股票，对应为stockPool的序号

        return index

    def trade(self):
        '''
        产生交易信号
        '''

        '''
        tradeLog = np.zeros(self.Cl.shape)

        for t in range(self.nTrain, self.T):
            print(t)
            # Train Model
            self.stockPool(t)
            stockIndex = self.trainModel(t)
            stks = set(np.array(self.stkPool)[stockIndex])  # 待交易的股票

            index = [i for i, s in enumerate(self.stks) if s in stks]

            # 置交易信号
            if t + self.nTrain - 1 < self.T:
                tradeLog[t+self.nTrain-1, index] = -1
            tradeLog[t, index] = 1

        self.tradeLog = tradeLog
        '''

        tradeLog = pd.DataFrame(np.zeros(self.Cl.shape), index=self.Cl.index, columns=self.Cl.columns)


        # TODO: 初始交易日
        for t in range(self.nTrain, self.T):


            if t % 20 == 0:
                print(t)
            # Train Model
            self.stockPool(t)
            index = self.trainModel(t)
            stocks = set(np.array(self.stkPool)[index])     # 待交易的股票
            # 设置交易信号
            tradeLog.loc[self.Cl.index[t], stocks] = 1

        self.tradeLog = tradeLog

    def run(self):
        self.trade()
        return self.tradeLog


if __name__ == '__main__':
    s = strategy(Cl, Op, Vol, mDate, stks, ListDate, nCluster=10)
    tradeLog = s.run()
