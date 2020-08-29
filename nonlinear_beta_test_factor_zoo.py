# %% Import Packages
import pandas as pd
import numpy as np
from scipy import linalg as LA
from scipy import stats as ST
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from myfunc import OLSRegression, kernel_test, loc_poly, my_bootstrap
import time
DATA_Folder = '/Users/lwg342/OneDrive - University of Cambridge/Utility/Data/Stock data/Factor Zoo Data/data/'
# %% [markdown]
# # Import Data
# 1. Return data. Here I use S&P500 constituents, can be extended to more stocks
# 2. Import Factor data from Factor Zoo
# %% Import and cleaning data
# RET = pd.read_csv("cleaned_RET.csv", index_col='date')
RET = pd.read_csv(
    DATA_Folder + "port202.csv", index_col=0, header=None)
RET.index = pd.to_datetime(RET.index, format='%Y%m')
RET.iloc[0:5, 0:5]

# %%
RET = pd.read_csv(
    DATA_Folder + "port_5x5.csv", header=None)
RET = RET.drop([0], axis=1)
RET.index = pd.date_range("1976-07-31", "2017-12-31", freq='M')
RET.iloc[0:5, 0:5]
# %% Import Factor Data
FACTOR = pd.read_csv(
    'factors_zoo.csv')
FACTOR.rename(columns={'  Date': 'date'}, inplace=True)
FACTOR.date = pd.to_datetime(FACTOR.date, format='%Y%m%d')
FACTOR = FACTOR.pivot_table(index='date')
FACTOR = FACTOR.dropna(axis=1)
FACTOR.iloc[0:5, 0:5]
# %% [markdown]
# # Individual test of nonlinearity
# %%
RET_excess = RET - np.array([FACTOR.RF]).T
FACTOR.RF.iloc[0:5]
RET_excess.iloc[0:5, 0:5]
average_ret = np.array(RET.mean())
average_ret_excess = np.array(RET_excess.mean())
# %% Apply to all factors
def nonlinear_beta_test(baseline_factor, FACTOR=FACTOR, RET_excess=RET_excess, with_intercept=1):
    Result = []
    tic = time.time()
    count = 0
    for i in FACTOR.columns[0:5]:
        if i in baseline_factor:
            Result.append(
                [i, '-----', '-----', '-----'])
        else:
            select = baseline_factor + [i]
            beta = np.array(OLSRegression(
                np.array(FACTOR[select]), RET_excess).beta_hat().iloc[:, 1:])
            Tn, Critical_left, Critical_right = my_bootstrap(
                beta, average_ret_excess, B=500, intercept=with_intercept)
            Result.append([i, Tn, Critical_left, Critical_right])
            if Tn < Critical_left or Tn > Critical_right:
                print("The factor", i, "is nonlinear!")
                count = count+1
    print("Total Elapsed Time =", time.time() - tic)
    print('Nonlinear Count =', count)
    return Result

# %% Different baseline factors.
# Individual
baseline_factor = []
Result = nonlinear_beta_test(baseline_factor)
pd.DataFrame(Result).to_csv('Result_individual.csv')
# %% Market As baseline
baseline_factor = ['MktRf']
[Tn, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Tn, Critical_left, Critical_right]
             ).to_csv('Result_baseline_mktrf.csv')
# %% FF3 as basline
baseline_factor = ['MktRf', 'HML', 'SMB']
[Tn, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Tn, Critical_left, Critical_right]
             ).to_csv('Result_baseline_ff3.csv')

# %% Taming the Factor Zoo chooses 4 Factors:
# SMB, nxf, chcsho, pm
baseline_factor = ['SMB', 'nxf', 'chcsho', 'pm']
[Tn, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Tn, Critical_left, Critical_right]
             ).to_csv('Result_baseline_TFZ4.csv')
# %% Taming the Factor Zoo chooses 4 Factors:
# SMB, nxf, chcsho, pm
baseline_factor = ['SMB', 'nxf', 'chcsho', 'pm']
[Tn, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Tn, Critical_left, Critical_right]
             ).to_csv('Result_baseline_TFZ4.csv')
# %%
result = pd.read_csv('result_individual_without_intercept.csv')
(result.iloc[0, 1:] < result.iloc[1, 1:]).sum()
(result.iloc[0, 1:] > result.iloc[2, 1:]).sum()
# %%
result = pd.read_csv('result_baseline_mktrf_without_intercept.csv')
(result.iloc[0, :] < result.iloc[1, :]).sum()
(result.iloc[0, :] > result.iloc[2, :]).sum()

# %%
result = pd.read_csv('result_baseline_ff3_without_intercept.csv')
(result.iloc[0, :] < result.iloc[1, :]).sum()
(result.iloc[0, :] > result.iloc[2, :]).sum()

# %%
result = pd.read_csv('result_baseline_TFZ4.csv')
(result.iloc[0, :] < result.iloc[1, :]).sum()
(result.iloc[0, :] > result.iloc[2, :]).sum()

# %%
j = 0
FNAME = list()
for i in FACTOR.columns:
    ff = i + ':::'
    FNAME.append(ff)
    print(ff)
FNAME = pd.DataFrame(FNAME)
FNAME.to_csv('FNAME.csv', index=False)


# %% Test on a specific model
beta = OLSRegression(
    np.array(FACTOR[['SMB', 'nxf', 'chcsho', 'pm']]), np.array(RET)).beta_hat()[:, 1:]
second_step_OLS = sm.OLS(sm.add_constant(beta), average_ret_excess).fit()
second_step_OLS.params
[a, b, c] = my_bootstrap(
    beta, average_ret_excess, B=500, intercept=1)
# %%
i = 3
plt.figure()
plt.scatter(beta[:, i], average_ret_excess)
plt.scatter(beta[:, i], OLSRegression(
    beta, average_ret_excess).y_hat(intercept=0))
plt.scatter(beta[:, i], OLSRegression(beta, average_ret_excess).y_hat())
# plt.scatter(beta[:,i], loc_poly(average_ret_excess, beta[:,i]))
print(a, b, c)

# %%
