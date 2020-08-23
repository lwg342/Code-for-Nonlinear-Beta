# %% Import Packages
import pandas as pd
import numpy as np
from scipy import linalg as LA
from scipy import stats as ST
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from myfunc import OLSRegression, kernel_test, loc_poly,my_bootstrap
import time
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
RET_excess = RET - np.array([FACTOR.RF]).T
Result = np.zeros(147)
Critical_left = np.zeros(147)
Critical_right = np.zeros(147)
average_ret = np.array(RET.mean())
average_ret_excess = np.array(RET_excess.mean())

# %% Test the procedure on one additional factors
j = 45
FACTOR.iloc[:, j]
beta = np.array(OLSRegression(
    np.array([FACTOR.MktRf, FACTOR.HML, FACTOR.SMB]).T, RET_excess).beta_hat().iloc[:, 1:])
# beta = np.array(OLSRegression(
#     np.array(FACTOR['MktRf']), RET_excess).beta_hat().iloc[:, 1:])
print(beta.shape)
m = OLSRegression(beta, average_ret_excess).y_hat(intercept=0)
m1 = OLSRegression(beta, average_ret_excess).y_hat(intercept=1)
gamma = OLSRegression(beta, average_ret_excess).beta_hat(intercept=0)
print(gamma)
plt.figure()
plt.xlim(0,2)
plt.ylim(0,0.02)
plt.scatter(beta[:, 0], m)
plt.scatter(beta[:, 0], m1)
plt.scatter(beta[:, 0], average_ret_excess)
plt.figure()
plt.xlim(-1,1)
plt.ylim(0,0.02)
plt.scatter(beta[:, 1], m)
plt.scatter(beta[:, 1], m1)
plt.scatter(beta[:, 1], average_ret_excess)
plt.figure()
plt.xlim(-1,1)
plt.ylim(0,0.02)
plt.scatter(beta[:, 2], m)
plt.scatter(beta[:, 2], m1)
plt.scatter(beta[:, 2], average_ret_excess)
a = my_bootstrap(beta, average_ret_excess)
b = my_bootstrap(beta, average_ret_excess, intercept=0)
print(a)
print(b)
# %% Apply to all factors 
def nonlinear_beta_test(baseline_factor, FACTOR = FACTOR, RET_excess = RET_excess):
    j = 0
    for i in FACTOR.columns[~FACTOR.columns.isin(baseline_factor)]:
        tic = time.time()
        select = baseline_factor + [i]
        beta = np.array(OLSRegression(
            np.array(FACTOR[select]), RET_excess).beta_hat().iloc[:, 1:])
        Result[j], Critical_left[j], Critical_right[j] = my_bootstrap(beta, average_ret_excess, B = 250, intercept= 1)
        print("Factor", i, "Elapsed Time =", time.time() - tic)
        if Result[j] < Critical_left[j] or Result[j] > Critical_right[j]:
            print("The factor", i, "is nonlinear!")
        j = j + 1
    return Result, Critical_left, Critical_right

# %% Different baseline factors.
# Individual
baseline_factor = []
[Result, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Result, Critical_left, Critical_right]
             ).to_csv('result_individual.csv')
# Market As baseline
baseline_factor = ['MktRf']
[Result, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Result, Critical_left, Critical_right]
             ).to_csv('result_baseline_mktrf.csv')
# FF3 as basline  
baseline_factor = ['MktRf', 'HML', 'SMB']
[Result, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Result, Critical_left,Critical_right]).to_csv('result_baseline_ff3.csv')

# %%
result = pd.read_csv('result_baseline_ff3.csv')
(result.iloc[0, :] < result.iloc[1,:]).sum()
(result.iloc[0, :] > result.iloc[2, :]).sum()
# %%
result = pd.read_csv('result_baseline_mktrf.csv')
(result.iloc[0, :] < result.iloc[1,:]).sum()
(result.iloc[0, :] > result.iloc[2, :]).sum()
