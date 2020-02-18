# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


class backtest:
    def __init__(self, Cl, Op, tradeLog, ratio=0.003, nTrain=20, nStock=5, initMoney=1e7):
        self.Cl = Cl                    # 收盘价（收盘时卖出）
        self.Op = Op                    # 开盘价（开盘时买入）
        self.ratio = ratio              # 手续费
        self.nTrain = nTrain            # 回望周期长度
        self.nStock = tradeLog.sum(axis = 1)            # 每期交易的股票数
        self.T = Cl.shape[0]            # 总交易时间
        self.nStks = Cl.shape[1]        # 股票总数

        self.nextPos = tradeLog.copy()  # 下一交易日开仓仓位
        self.Pos = pd.DataFrame(np.zeros(Cl.shape), index=Cl.index, columns=Cl.columns)  # 每只股票的持仓记录

        self.Ret = np.zeros(self.T)     # 记录总资产
        self.Cash = np.zeros(self.T)    # 记录手头资金
        self.Equity = np.zeros(self.T)  # 记录持仓股票价值
        self.Ret[:(nTrain + 1)] = initMoney
        self.Cash[:(nTrain + 1)] = initMoney

    def sharpe_ratio(self):
#    '''夏普比率
#    '''
        df = self.Ret
    #    df = pd.DataFrame(return_list)
        total_ret=df[-1]-1
        annual_ret=pow(1+total_ret,12/len(df))-1
        annual_std = df.std()
        sharpe_ratio = (annual_ret-0.02) * np.sqrt(12)  / annual_std  #默认12个月,无风险利率为0.02
        return annual_ret,sharpe_ratio 
    
    
    
    def info_ratio(self):
#        '''信息比率
#        '''
        df = self.Ret
    #    df = pd.DataFrame(return_list)
        total_ret=df[-1]-1
        annual_ret=pow(1+total_ret,12/len(df))-1
        annual_std = df.std()
        info_ratio = (annual_ret+0.005) * np.sqrt(12)  / annual_std  #默认12个月,减去HSI。。。
        return info_ratio 
    
    
    def MaxDrawdown(self):
#        '''最大回撤率
#        '''
        a = self.Ret
        i = np.argmax((np.maximum.accumulate(a) - a) / np.maximum.accumulate(a))  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(a[:i])  # 开始位置
        return (a[j] - a[i]) / (a[j])


    def trade(self):


        # TODO: 统计
        winTimes, loseTimes, profit, loss = 0, 0, 0, 0


        enterCash = self.Ret[0] / self.nTrain     # 前nTrain日，每日开仓资金

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


    '''
    def trade(self):

        enterPrice = np.zeros(self.nStks)

        # 统计
        wintimes, losetimes, profit, loss = 0,0,0,0

        for t in range(1,self.T):
            # 初始化
            print(t)
            self.cash[t] = self.cash[t-1]
            self.Equity[t] = self.Equity[t-1]

            numOfBuys = sum(self.tradeLog[t,:]==1)   # 待买股票数量
            buymoney = 0.4*self.cash[t]/numOfBuys if numOfBuys!=0 else 0

            for s in range(self.nStks):
                tlog = self.tradeLog[t,s]  # 当前信号
                pos = self.Pos[t-1,s]      # 昨日持仓
                curPrice = self.Op[t,s]
                yestPrice = self.Op[t-1,s]

                if tlog == 1 and pos == 0:  # 开仓
                    enterPrice[s] = curPrice   # 开仓价格
                    self.Pos[t,s] = np.floor(buymoney/enterPrice[s])
                    self.cash[t] -= self.Pos[t,s]*enterPrice[s]*(1+self.ratio)
                    self.Equity[t] += self.Pos[t,s]*enterPrice[s]

                elif tlog == -1 and pos != 0:  # 平仓
                    self.cash[t] += curPrice*pos
                    self.Equity[t] -= yestPrice*pos
                    self.Pos[t, s] = 0

                    # 统计
                    if curPrice > enterPrice[s]*(1+self.ratio):
                        wintimes += 1
                        profit += pos*(curPrice-enterPrice[s]*(1+self.ratio))
                    else:
                        losetimes += 1
                        loss += pos*(curPrice-enterPrice[s]*(1+self.ratio))

                else:
                    self.Equity[t] += pos*(curPrice-yestPrice)
                    self.Pos[t, s] = pos

        self.wintimes, self.losetimes, self.profit, self.loss = wintimes, losetimes, profit, loss
        self.Ret = self.cash + self.Equity
        return self.Ret, self.Pos
        '''


if __name__ == '__main__':
    bt = backtest(Cl, Op, tradeLog, ratio=0.003, initMoney=1e7)
    Ret, Pos,Cash,nStock = bt.trade()
    s_r = bt.sharpe_ratio()
    s_ir = bt.info_ratio()
    s_mdd = bt.MaxDrawdown()  
    
