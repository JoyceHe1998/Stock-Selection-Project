# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class backtest:
    def __init__(self, Cl, Op, tradeLog, ratio=0.003, nTrain=20, nStock=5, initMoney=1e7):
        self.Cl = Cl                    # Closing price
        self.Op = Op                    # Opening price
        self.ratio = ratio              # Transaction fee
        self.nTrain = nTrain            
        self.nStock = tradeLog.sum(axis = 1)            # Number of stocks to trade
        self.T = Cl.shape[0]            # Transaction time
        self.nStks = Cl.shape[1]        # Total number of stocks

        self.nextPos = tradeLog.copy()  
        self.Pos = pd.DataFrame(np.zeros(Cl.shape), index=Cl.index, columns=Cl.columns)  

        self.Ret = np.zeros(self.T)    
        self.Cash = np.zeros(self.T)   
        self.Equity = np.zeros(self.T)  
        self.Ret[:(nTrain + 1)] = initMoney
        self.Cash[:(nTrain + 1)] = initMoney

    def sharpe_ratio(self):
        df = self.Ret
        total_ret=df[-1]-1
        annual_ret=pow(1+total_ret,12/len(df))-1
        annual_std = df.std()
        sharpe_ratio = (annual_ret-0.02) * np.sqrt(12)  / annual_std 
        return annual_ret,sharpe_ratio 
    
    
    
    def info_ratio(self):
        df = self.Ret
        total_ret=df[-1]-1
        annual_ret=pow(1+total_ret,12/len(df))-1
        annual_std = df.std()
        info_ratio = (annual_ret+0.005) * np.sqrt(12)  / annual_std
        return info_ratio 
    
    
    def MaxDrawdown(self):
        a = self.Ret
        i = np.argmax((np.maximum.accumulate(a) - a) / np.maximum.accumulate(a)) 
        if i == 0:
            return 0
        j = np.argmax(a[:i])  
        return (a[j] - a[i]) / (a[j])


    def trade(self):
        winTimes, loseTimes, profit, loss = 0, 0, 0, 0
        enterCash = self.Ret[0] / self.nTrain 

        for t in range(1 + self.nTrain, 2 * self.nTrain):
            self.nextPos.iloc[t - 1] *= enterCash / self.nStock[t] / (1 + self.ratio) / self.Op.iloc[t]
            if self.nStock[t] == 0:
                self.nextPos.iloc[t - 1] = 0
            self.Pos.iloc[t] = self.Pos.iloc[t - 1] + self.nextPos.iloc[t - 1]
            self.Equity[t] = np.dot(self.Pos.iloc[t], self.Cl.iloc[t])
            self.Cash[t] = self.Cash[t - 1] - enterCash

        for t in range(2 * self.nTrain, self.T):
            self.nextPos.iloc[t - 1] *= self.Cash[t - 1] / self.nStock[t] / (1 + self.ratio) / self.Op.iloc[t]
            if self.nStock[t] == 0:
                self.nextPos.iloc[t - 1] = 0
            self.Pos.iloc[t] = self.Pos.iloc[t - 1] + self.nextPos.iloc[t - 1] - self.nextPos.iloc[t - self.nTrain]
            self.Equity[t] = np.dot(self.Pos.iloc[t], self.Cl.iloc[t])
            self.Cash[t] = np.dot(self.nextPos.iloc[t - self.nTrain], self.Cl.iloc[t]) * (1 - self.ratio)
            print(t,self.nStock[t],)
        self.Ret = (self.Equity + self.Cash)/1e7
        return self.Ret, self.Pos,self.Cash,self.nStock

if __name__ == '__main__':
    bt = backtest(Cl, Op, tradeLog, ratio=0.003, initMoney=1e7)
    Ret, Pos,Cash,nStock = bt.trade()
    s_r = bt.sharpe_ratio()
    s_ir = bt.info_ratio()
    s_mdd = bt.MaxDrawdown()  
    
