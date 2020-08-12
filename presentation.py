# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from scipy import linalg as LA
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.distributions.empirical_distribution import ECDF

# %% [markdown]
#  ## Data
#  The data is from CRSP beta-suite, consisting of daily data of 253 stocks, all of which are constituents of SP500, from 1990-01-02 to 2019-12-31. The factors are FF 3 factors plus Carhart's factor.

# %%
T = pd.read_csv(
    "/Users/lwg342/Dropbox/Cam Econ Trio/Data/cleaned_sp500_1990_2019_4factors_beta_suite.csv", index_col='PERMNO')
print(T.head())
T.DATE = pd.to_datetime(T.DATE)

# %% [markdown]
#  In the next cell we create the matrices for returns, excess returns and idiosyncratic volatility.

# %%
vol = pd.pivot_table(T, 'ivol', index='PERMNO', columns='DATE')
tvol = pd.pivot_table(T, 'tvol', index='PERMNO', columns='DATE')
exret = pd.pivot_table(T, 'exret', index='PERMNO', columns='DATE')
ret = pd.pivot_table(T, 'RET', index='PERMNO', columns='DATE')

# %% [markdown]
 ## Calculating covariance matrices
#  Assume that we have a factor pricing model:
#  $R_{it} = a_i + b_i' f_t  + e_{it}$
#  In the next cell, we calculate the sample covariance matrices of the variables involved. 
#  We use a non-overlapping rolling window, the window width is 6 month.
#  So $C_t = var(R_t)$, for t being each 6 month window. $R_t$ is the N-dim return vector

# %%
# 3: C = cov(ret)
C = pd.pivot_table(T, 'RET', index='DATE', columns='PERMNO').groupby(
    pd.Grouper(freq='180D')).apply(lambda x: np.cov(x, rowvar=False))

# %% [markdown]
#  Once we have the covariance matrix, we calculate the eigenvalues of the covariance matrices for each window.

# %%
C_eigen = np.stack(C.apply(lambda x: np.flip(LA.eigh(x, eigvals_only=True))))

# %% [markdown]
#  ## Plots
# %% [markdown]
#  ### Empirical CDF of log(eigenvalues)
#  Let $\lambda_j, j=1,...,N$ ,where $N = 253$, be the eigenvalues of the covariance matrix, in descending order, the next figure plots the empirical CDF of $ \log(\lambda_j)$ for each window.

# %%
plt.figure()
for j in range(0, C_eigen.shape[0]):
    ecdf = ECDF(np.log(C_eigen[j, :]))
    plt.xlim(-10, -3)
    plt.ylim(0.2, 0.8)
    plt.plot(ecdf.x, ecdf.y)

# %% [markdown]
#  ### The time series of specific eigenvalues
#  In the next figure, I plot the time series of the 6 largest eigenvalues together with roughly 20%, 40%, 60%, 80%-quantile eigenvalues.
#  It shows a similarity in pattern

# %%
plt.figure()
plt.ylim(-10, 0)
plt.plot(C.index, np.log(C_eigen[:, 0]))
plt.plot(C.index, np.log(C_eigen[:, 1]))
plt.plot(C.index, np.log(C_eigen[:, 2]))
plt.plot(C.index, np.log(C_eigen[:, 3]))
plt.plot(C.index, np.log(C_eigen[:, 4]))
plt.plot(C.index, np.log(C_eigen[:, 5]))
plt.plot(C.index, np.log(C_eigen[:, 50]))
plt.plot(C.index, np.log(C_eigen[:, 100]))
plt.plot(C.index, np.log(C_eigen[:, 150]))
plt.plot(C.index, np.log(C_eigen[:, 200]))
plt.plot(C.index, np.log(C_eigen[:, 250]))

# %% [markdown]
#  ### Plot of the ACF of the time series of $\log(\lambda_1)$.

# %%
plot_acf(np.log(C_eigen[:, 0]))
