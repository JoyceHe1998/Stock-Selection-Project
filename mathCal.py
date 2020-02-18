# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib


# 1. 计算ROC
def calROC(Cl, period):
    '''
    计算 Rate of Change
    :return: roc 为与Cl同样大小的矩阵
    '''
    ClV = Cl.values
    roc = np.zeros(Cl.shape)
    for t in range(period, Cl.shape[0]):
        roc[t, :] = (ClV[t, :] - ClV[t - period, :]) / ClV[t - period, :] * 100
    roc = pd.DataFrame(roc, index=Cl.index, columns=Cl.columns)
    return roc


# 2. 计算偏离值
def calTrend(Cl, period):
    '''
    计算 Trend 指标
    :return: trend 为与Cl同样大小的矩阵
    '''
    # ema = pd.ewma(Cl, span=period)
    ema = Cl.ewm(span=period).mean()
    return (Cl - ema) / ema * 100


#另外加 
# RSI里Cl只能是一列，所以模仿上面加个for-loop，用的是talib库里的method     
def calRSI(Cl, period):
    
    ClV = Cl.values
    rsi = np.zeros(Cl.shape)
    for t in range(0, Cl.shape[1]):
        rsi[:,t] = talib.RSI(ClV[:,t], timeperiod=6)
#        addded
    rsi = pd.DataFrame(rsi, index=Cl.index, columns=Cl.columns)
    rsi.fillna(0, inplace=True)
#    rsi = talib.RSI(Cl.values, timeperiod=6)
    return rsi


if __name__ == '__main__':
    roc = calROC(Cl, period=20)
    trend = calTrend(Cl, period=50)
    rsi = calRSI(Cl, period = 6)

