# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib

def calROC(Cl, period):
    ClV = Cl.values
    roc = np.zeros(Cl.shape)
    for t in range(period, Cl.shape[0]):
        roc[t, :] = (ClV[t, :] - ClV[t - period, :]) / ClV[t - period, :] * 100
    roc = pd.DataFrame(roc, index=Cl.index, columns=Cl.columns)
    return roc

def calTrend(Cl, period):
    ema = Cl.ewm(span=period).mean()
    return (Cl - ema) / ema * 100
 
def calRSI(Cl, period):
    
    ClV = Cl.values
    rsi = np.zeros(Cl.shape)
    for t in range(0, Cl.shape[1]):
        rsi[:,t] = talib.RSI(ClV[:,t], timeperiod=6)
    rsi = pd.DataFrame(rsi, index=Cl.index, columns=Cl.columns)
    rsi.fillna(0, inplace=True)
    return rsi


if __name__ == '__main__':
    roc = calROC(Cl, period=20)
    trend = calTrend(Cl, period=50)
    rsi = calRSI(Cl, period = 6)

