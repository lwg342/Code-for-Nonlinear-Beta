# %% Import Packages
import pandas as pd
import numpy as np
from scipy import linalg as LA
from scipy import stats as ST
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from myfunc import OLS_mean, kernel_test, loc_poly,my_bootstrap
# %% OLS parameter estimate
def OLS_parameter(X, Y):
    beta = sm.OLS(Y, sm.add_constant(X)).fit().params.T
    return beta
# %% [markdown]
# # Import Data
# 1. Return data. Here I use S&P500 constituents, can be extended to more stocks
# 2. Import Factor data from Factor Zoo
# %% Import and cleaning data
DATA = pd.read_csv(
    'HPRET_monthly_correspondint_to_factor_zoo_all.csv')
DATA.date = pd.to_datetime(DATA.date, format='%Y%m%d')

DATA = DATA.drop(DATA[DATA.RET == 'C'].index)
DATA = DATA.drop(DATA[DATA.RET == 'B'].index)
DATA.RET = DATA['RET'].astype('float')
RET = DATA.pivot_table('RET', index='PERMNO', columns='date')
RET = RET.dropna(0).transpose()
RET.iloc[0:5,0:5]
# %% Import Factor Data
FACTOR = pd.read_csv(
    'factors_zoo.csv')
FACTOR.rename(columns={'  Date': 'date'}, inplace=True)
FACTOR.date = pd.to_datetime(FACTOR.date, format='%Y%m%d')
FACTOR = FACTOR.pivot_table(index = 'date')
FACTOR = FACTOR.dropna(axis=1)
FACTOR.iloc[0:5,0:5]
# %% [markdown] 
# # Individual test of nonlinearity
# %%
Result = np.zeros(147)
Critical_left = np.zeros(147)
Critical_right = np.zeros(147)
average_ret = np.array(RET.mean())
import time
tic = time.time()
for i in range(0,147):
    beta = np.array(OLS_parameter(FACTOR.iloc[:,i], RET).iloc[:, 1])
    # average_ret = np.array(RET.iloc[8,:])
    # plt.figure
    # plt.scatter(beta, average_ret)
    Result[i], Critical_left[i], Critical_right[i] = my_bootstrap(beta, average_ret)
    print(i)
elapsed = time.time() - tic
pd.DataFrame([Result, Critical_left,Critical_right]).to_csv('result.csv')
print(elapsed)

# %%
result = pd.read_csv('result.csv')
(result.iloc[0, :] < result.iloc[1,:]).sum()
(result.iloc[0, :] > result.iloc[2, :]).sum()

# %%
