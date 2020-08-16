#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:03:00 2020
@author: lwg342
"""
# %% import required libraries
import pandas as pd
import numpy as np
from scipy import linalg as LA
from scipy import stats as ST
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
# %% [markdown] 
# ## Data 
# The data is from CRSP beta-suite, consisting of daily data of 253 stocks, all of which are constituents of SP500, from 1990-01-02 to 2019-12-31. The factors are FF 3 factors plus Carhart's factor. 


# %% import dataset
T = pd.read_csv(
    "/Users/lwg342/Dropbox/Cam Econ Trio/Data/cleaned_sp500_1990_2019_4factors_beta_suite.csv", index_col='PERMNO')
print(T.head())
T.columns
T.DATE = pd.to_datetime(T.DATE)
date = T['DATE'].unique()
name = T.index.unique()

# %% [markdown] 
# In the next cell we create the matrices for returns, excess returns and idiosyncratic volatility. 
# %%  Create the matrices    
# idio volatility
vol = pd.pivot_table(T, 'ivol', index='PERMNO', columns = 'DATE')
# vol = vol.dropna(0)
# total volatility
tvol = pd.pivot_table(T, 'tvol', index='PERMNO', columns = 'DATE')
# tvol = tvol.dropna(0)
# excesss returns
exret= pd.pivot_table(T, 'exret', index = 'PERMNO',columns = 'DATE')
ret= pd.pivot_table(T, 'RET', index = 'PERMNO',columns = 'DATE')

# %% [markdown] 
# ## Calculating covariance matrices
# Assume that we have a factor pricing model:
# $$R_{it} = a_i + b_i' f_t  + e_{it}$$
# In the next cell, we calculate the sample covariance matrices of the variables involved.
# We use a non-overlapping rolling window, the window width is 6 month. 
# So $C_t = var(R_t)$, for t being each 6 month window. $R_t$ is the N-dim return vector
# %% Create rolling-window estimate of the covariance matrices
# Let e:= exret 
# 1: C = cov(e^2)
# C = (exret**2).groupby(|exret.columns.year, axis=1).apply(np.cov)
# 2: C = cov(e) 
# C = (exret).groupby(exret.columns.year, axis=1).apply(np.cov)
# 3: C = cov(ret)
# C = pd.pivot_table(T,'RET', index = 'DATE',columns = 'PERMNO').groupby(pd.Grouper(freq='180D')).apply(lambda x: np.cov(x,rowvar= False))
# 4: rolling window: this is not very efficient
C = pd.pivot_table(T,'RET', index = 'DATE',columns = 'PERMNO').rolling(window = '100D').apply(lambda x: np.cov(x,rowvar= False)).dropna(0)

# %% [markdown] 
# Once we have the covariance matrix, we calculate the eigenvalues of the covariance matrices for each window.
# %% PCA analysis for the excess returns
C_eigen = np.stack(C.apply(lambda x: np.flip(LA.eigh(x, eigvals_only=True))))

# %% Plot the log of all the eigenvalues
# plt.figure()
# for j in np.arange(0,5):
# plt.plot(np.log(C_eigen[:]))

# %% [markdown]
# ## Plots
# %% [markdown]
# ### Empirical CDF of eigenvalues
# Let $\lambda_j, j=1,...,N$ ,where $N = 253$, be the eigenvalues of the covariance matrix, in descending order, the next figure plots the empirical CDF of $ \log(\lambda_j)$ for each window. 
# %% Plot the CDF of log-eigenvalues over time
plt.figure()
for j in range(0,C_eigen.shape[0]):
    ecdf = ECDF(np.log(C_eigen[j, :]))
    # plt.xlim(-10, -3)
    # plt.ylim(0.2,0.8)
    plt.plot(ecdf.x,ecdf.y)

# %% [markdown]
# In the next figure, I plot the time series of the 6 largest eigenvalues together with roughly 20%, 40%, 60%, 80%-quantile eigenvalues. 
# It shows a similarity in pattern 
# %% plot the dynamic of the quantiles
plt.figure()
plt.ylim(-10,0)
plt.plot(C.index,np.log(C_eigen[:,0]))
plt.plot(C.index,np.log(C_eigen[:,1]))
plt.plot(C.index,np.log(C_eigen[:,2]))
plt.plot(C.index,np.log(C_eigen[:,3]))
plt.plot(C.index,np.log(C_eigen[:,4]))
plt.plot(C.index,np.log(C_eigen[:,5]))
plt.plot(C.index,np.log(C_eigen[:,50]))
plt.plot(C.index,np.log(C_eigen[:,100]))
plt.plot(C.index,np.log(C_eigen[:,150]))
plt.plot(C.index,np.log(C_eigen[:,200]))
plt.plot(C.index,np.log(C_eigen[:,250]))

# %% [markdown]
# I plot the ACF of the time series of $\log(\lambda_1)$.
# %% ACF of the 
plot_acf(np.log(C_eigen[:, 30])); print()
# def autocorr(x, t=2):
#     return np.corrcoef(np.array([x[:-t], x[t:]]))
# autocorr(np.log(C_eigen[:, 30]))
# %% As a comparison, we consider randomly generated matrices
for j in range(0,30):
    C_eigen = np.stack(np.flip(
        LA.eigh(np.cov(np.random.normal(size=[253, 365])), eigvals_only=True)))
    ecdf = ECDF(np.log(C_eigen))
    plt.xlim(-1, 0)
    plt.ylim(0.3,0.4)
    plt.plot(ecdf.x, ecdf.y)

# %% Now we let there be heteroscedasticity

for j in range(0,30):
    temp_matrix = np.abs(0.05*np.diag(np.random.normal(size=253))
                     )@np.random.normal(size=[253, 365])
    C_eigen = np.stack(np.flip(LA.eigh(np.cov(temp_matrix), eigvals_only = True)))
    ecdf = ECDF(np.log(C_eigen))
    # plt.xlim(0, 5)
    # plt.ylim(0, 0.6)
    plt.plot(ecdf.x, ecdf.y)

# %% [markdown] 
# To make comparison even clearer, we plot the two empirical CDFs together
# Here we have two features: 1. Heteroscedasticiy across i; 2. time-varying volatility
# %%
C_eigen = np.stack(C.apply(lambda x: np.flip(LA.eigh(x, eigvals_only=True))))
plt.figure()
for j in range(0, C_eigen.shape[0]):
    ecdf = ECDF(np.log(C_eigen[j, 0:100]))
    # plt.xlim(-10, -3)
    # plt.ylim(0.2,0.8)
    plt.plot(ecdf.x, ecdf.y)

for j in range(0, 30):
    temp_matrix = np.abs(np.random.normal(scale = 0.01))*np.abs(1*np.diag(np.random.normal(size=253))
                         )@np.random.normal(size=[253, 365])
    C_eigen = np.stack(
        np.flip(LA.eigh(np.cov(temp_matrix), eigvals_only=True)))
    ecdf = ECDF(np.log(C_eigen[0:100]))
    # plt.xlim(0, 5)
    # plt.ylim(0, 0.6)
    plt.plot(ecdf.x, ecdf.y)

# %% Histogram

plt.figure()
# for j in range(3):
plt.hist(np.log(C_eigen[14,:]), bins = 30)

# %% Eigen-decomposition of the matrix C
E = np.linalg.eig(C.iloc[0])
Lamb = E[1][:,0]



# %% [markdown] 
# Now we try the empirical distribution of the eigenvalues for a rolling window
# The rolling window frequency is Monthly

# %%
C1 = pd.pivot_table(T, 'RET', index='DATE', columns='PERMNO').groupby(
    pd.Grouper(freq='60D')).apply(lambda x: np.cov(x, rowvar= False))
# C1 = pd.pivot_table(T, 'RET', index='DATE', columns='PERMNO').groupby(
#     pd.Grouper(freq='60D')).apply(lambda x: np.transpose(x)@x)
# %%
E_value = np.zeros([name.size, C1.size])
for t in np.arange(0,C1.size):
    E_value[:, t] = np.flip(LA.eigh(C1.iloc[t], eigvals_only=True))

# %% Plot the dynamic of the eigenvalues. Taking Log
for j in np.arange(0,5):
    plt.plot(C1.index, np.log(E_value[j,:]))

# %% Level
plt.figure()
# for j in np.arange(0,5):
    # plt.plot(C1.index, (E_value[j,:]))
plt.plot(C1.index, np.log(E_value[0,:]))

# %%
plot_acf(exret.iloc[5,:]**2)

# %% Some plots 
# plt.figure()
# We plot the 101th stock's time series of excess return
# plt.plot(exret.columns, exret.iloc[100,:]**2)

# %% Plot exret against beta
pltx = T.pivot_table('b_smb', index='DATE', columns='PERMNO').groupby(
    pd.Grouper(freq='180D')).mean()
plty = T.pivot_table('exret', index='DATE', columns='PERMNO').groupby(
    pd.Grouper(freq='180D')).mean()
# %% Nonparametric regression
for t0 in range(0,30):
    plt.figure()
    loc_poly(np.array(plty.iloc[t0, :]), np.array(pltx.iloc[t0, :]))
    plt.legend()

# %% 
T[T.DATE == date[t0]].plot.scatter(x='b_mkt', y='exret')
T[T.DATE == date[t0]].plot.scatter(x='b_smb', y='exret')
T[T.DATE == date[t0]].plot.scatter(x='b_hml', y='exret')
# %%
T[T.DATE == date[10]].plot.scatter(x='b_mkt', y='exret')
T[T.DATE == date[120]].plot.scatter(x='b_mkt', y='ivol')

# %%
t0 = 600
X = np.array([T[T.DATE == date[t0]].b_mkt, T[T.DATE == date[t0]].b_hml]).T
Y = np.array(T[T.DATE == date[t0]].exret)
m = OLS_mean(X,Y)
u = Y - m
Test_0 = kernel_test(u, X)
plt.scatter(np.array(T[T.DATE == date[t0]].b_mkt),
            np.array(T[T.DATE == date[t0]].exret))
result = my_bootstrap(X, Y)

# %%
