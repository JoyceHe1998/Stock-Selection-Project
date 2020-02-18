# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


class drawPicture:
    def __init__(self, Ret, Pos, mDate,Cl,start_date, end_date):
        self.Ret = Ret / Ret[0]
        self.Pos = Pos
        self.mdate = mDate
        self.Cl = Cl
        self.start_date = start_date
        self.end_date = end_date
        pass

    def loadIndexData(self):
        path = r'C:\Users\heyiz\Desktop\python_notes\new\Data\HSI.xlsx'
        df = pd.read_excel(path,index_col=0)
#        self.SH01 = df['上证综指'].values
        self.HSI = df['Close']
#        self.SH01 = self.SH01 / self.SH01[0]
        
#        self.HSI.index = [datetime.datetime.strftime(x, '%Y-%m-%d') for x in self.HSI.index]
        print(self.Cl.index[0])
#        print('2007-01-31 00:00:00' in self.HSI )

        index_date_range = []

        for i in self.HSI.index:
            time = i
            if (( time >= datetime.datetime.strptime(self.start_date,'%Y-%m-%d')) & (time <= datetime.datetime.strptime(self.end_date,'%Y-%m-%d'))):            
                index_date_range.append(i)
        self.HSI = self.HSI[index_date_range]

#        self.HSI  = self.HSI[self.Cl.index]

        
        print(self.HSI)
        self.HSI = self.HSI / self.HSI[0]
        self.date = df.index

    def drawRet(self):
        self.loadIndexData()

        plt.figure(figsize=(8, 5))
        
        plt.plot(self.Cl.index, self.Ret, label='Ret')
#        plt.plot(self.mdate, self.SH01, label='SH000001')
#        print(self.mdate)
#        print(self.HS300)
        plt.plot(self.HSI.index, self.HSI, label='HSI')
        plt.xlabel('Time')
        plt.ylabel('Rate')
        plt.legend()

#        plt.figure(figsize=(8, 5))
#        plt.plot(self.mdate, self.Ret - self.SH01, label='Hedge')
#        plt.xlabel('Time')
#        plt.ylabel('Rate')
#        plt.legend()


if __name__ == '__main__':
    d = drawPicture(Ret, Pos, mDate,Cl,start_date,end_date)
    d.drawRet()
