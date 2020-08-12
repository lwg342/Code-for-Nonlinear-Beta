#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:03:00 2020

@author: lwg342
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

T= pd.read_csv("/Users/lwg342/OneDrive - University of Cambridge/Utility/Data/Stock data/cleaned_sp500_1990_2019_4factors_beta_suite.csv",index_col='PERMNO')
T.head()
T.columns
date = T['DATE'].unique()
name = T.index.unique()


# idio volatility
vol = pd.pivot_table(T, 'ivol', index='PERMNO', columns = 'DATE')
vol = vol.dropna(0)

# total volatility

tvol = pd.pivot_table(T, 'tvol', index='PERMNO', columns = 'DATE')
tvol = tvol.dropna(0)

diff_vol = tvol - vol
ratio_diff = (vol/tvol)**2
R2 = pd.pivot_table(T,'R2',index= 'PERMNO',columns= 'DATE')


# excesss returns

exret= pd.pivot_table(T, 'exret', index = 'PERMNO',columns = 'DATE')


# def  = my_rolling_pca()

C = exret.groupby(exret.columns.year,axis = 1).apply(np.cov)
pca = PCA(n_components=5)
pca.fit(C)
print(pca.explained_variance_ratio_)

E = np.linalg.eig(C)
Lamb = E[1][:,0]



# plots


vol.iloc[100].plot()
pick_date = date[1500]

T[T.DATE == date[10]].plot.scatter(x = 'b_mkt',y = 'exret')
T[T.DATE == pick_date].plot.scatter(x = 'b_mkt',y = 'RET')
T[T.DATE == pick_date].plot.scatter(x = 'b_smb',y = 'RET')
T[T.DATE == pick_date].plot.scatter(x = 'b_hml',y = 'RET')
T[T.DATE == date[120]].plot.scatter(x = 'b_mkt',y = 'ivol')

