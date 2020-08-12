# %% import required libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg as LA
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.distributions.empirical_distribution import ECDF

# %% [markdown] 
# # Data Description
# The data is from CRSP beta-suite, consisting of daily data of 253 stocks, all of which are constituents of SP500, from 1990-01-02 to 2019-12-31. The factors are FF 3 factors plus Carhart's factor. 

# %% import dataset
T = pd.read_csv(
    "/Users/lwg342/Dropbox/Cam Econ Trio/Data/cleaned_sp500_1990_2019_4factors_beta_suite.csv", index_col='PERMNO')
T.columns
T.DATE = pd.to_datetime(T.DATE)
date = T['DATE'].unique()
name = T.index.unique()
T.head()
# %% [markdown] In the next cell we create the matrices for returns, excess returns and idiosyncratic volatility.
# %%  Create the matrices    
# idio volatility
vol = pd.pivot_table(T, 'ivol', index='PERMNO', columns = 'DATE')
# total volatility
tvol = pd.pivot_table(T, 'tvol', index='PERMNO', columns = 'DATE')
# excesss returns
exret= pd.pivot_table(T, 'exret', index = 'PERMNO',columns = 'DATE')
# return
ret= pd.pivot_table(T, 'RET', index = 'PERMNO',columns = 'DATE')

# %% [markdown] 
# # Rolling-Window Estimating Covariance Matrices
# We use 6-month non-overlapping rolling windows to estimate the $N\times N$ sample covariance matrices. 
# %%
C = pd.pivot_table(T,'RET', index = 'DATE',columns = 'PERMNO').groupby(pd.Grouper(freq='6M')).apply(lambda x: np.cov(x,rowvar= False))

# %% [markdown]
# # Eigen-decomposition of each covariance matrices. 
# %%
C_eigen = np.stack(C.apply(lambda x: np.flip(LA.eigh(x, eigvals_only=True))))

# %% [markdown]
# Plots 
# ## CDF of log(eigenvalues)
# %% Plot the CDF of log-eigenvalues over time
plt.figure()
for j in range(0,C_eigen.shape[0]):
    ecdf = ECDF(np.log(C_eigen[j, :]))
    # plt.xlim(-8, -6)
    # plt.ylim(0.4,0.6)
    plt.plot(ecdf.x,ecdf.y)

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

# %% ACF of the median
plot_acf(np.log(C_eigen[:, 100]))
np.corrcoef(C_eigen[:,[0,10]])
# %% As a comparison, we consider randomly generated matrices
for j in range(0,30):
    C_eigen = np.stack(np.flip(np.log(
        LA.eigh(np.cov(np.random.normal(size=[253, 365])), eigvals_only=True))))
    ecdf = ECDF(np.log(C_eigen))
    plt.xlim(-3, -1)
    plt.ylim(0,0.15)
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

# %%  plots
vol.iloc[100].plot()
pick_date = date[1500]
T[T.index == name[1]].plot.scatter(x='DATE', y='exret')
T[T.DATE == date[10]].plot.scatter(x='b_mkt', y='exret')
T[T.DATE == pick_date].plot.scatter(x='b_mkt', y='RET')
T[T.DATE == pick_date].plot.scatter(x='b_smb', y='RET')
T[T.DATE == pick_date].plot.scatter(x='b_hml', y='RET')
T[T.DATE == date[120]].plot.scatter(x='b_mkt', y='ivol')