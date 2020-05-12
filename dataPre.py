# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime


class dataPre:

    def __init__(self):
        pass

    def loadDataFromCsv(self):
        path = r'C:\Users\heyiz\Desktop\python_notes\yahoo_clean_Close.csv'
        Cl = pd.read_csv(path,index_col = 0)
        print(Cl)
        
        start_date = '2007-01-01'
        end_date = '2017-01-01'
    
        Cl.replace({0: np.nan}, inplace=True)
        Cl.fillna(method='ffill', inplace=True)
        Cl.fillna(method='bfill', inplace=True)
        Cl.fillna(1, inplace=True)
        Cl.index =  [datetime.datetime.strptime(x, '%Y-%m-%d') for x in Cl.index]
        Cl1 = Cl.resample('M').last()
        
        index_date_range = []
        
        for i in Cl1.index:
            time = i
            if (( time >= datetime.datetime.strptime(start_date,'%Y-%m-%d')) & (time <= datetime.datetime.strptime(end_date,'%Y-%m-%d'))):            
                index_date_range.append(i)
        Cl = Cl1.loc[index_date_range,:]

        path = r'C:\Users\heyiz\Desktop\python_notes\yahoo_clean_Open.csv'
        Op = pd.read_csv(path,index_col = 0)
        Op.replace({0: np.nan}, inplace=True)
        Op.fillna(method='ffill', inplace=True)
        Op.fillna(method='bfill', inplace=True)
        Op.fillna(1, inplace=True)
        Op.index =  [datetime.datetime.strptime(x, '%Y-%m-%d') for x in Op.index]
        Op = Op.resample('M').first()
        Op.index = Cl1.index
        
        Op = Op.loc[index_date_range,:]

        path = r'C:\Users\heyiz\Desktop\python_notes\yahoo_clean_Volume.csv'
        Vol = pd.read_csv(path, index_col = 0)
        Vol.index =  [datetime.datetime.strptime(x, '%Y-%m-%d') for x in Vol.index]
        Vol = Vol.resample('M').sum()
        Vol.index = Cl1.index
        Vol = Vol.loc[index_date_range,:]
        self.Cl, self.Op, self.Vol, self.start_date, self.end_date = Cl, Op, Vol, start_date, end_date
        self.mDate = pd.DataFrame(list(Cl.index))
        self.stks = pd.DataFrame(list(Cl.columns))

        return self.Cl, self.Op, self.Vol, self.mDate, self.stks,self.start_date, self.end_date

    def loadListDate(self, filename='ListingDate'):
        return pd.DataFrame()


if __name__ == '__main__':
    Cl, Op, Vol, mDate, stks, start_date, end_date = dataPre().loadDataFromCsv()
    ListDate = dataPre().loadListDate()
