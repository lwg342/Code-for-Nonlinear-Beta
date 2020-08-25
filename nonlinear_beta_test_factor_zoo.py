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
# %% [markdown]
# # Import Data
# 1. Return data. Here I use S&P500 constituents, can be extended to more stocks
# 2. Import Factor data from Factor Zoo
# %% Import and cleaning data
RET = pd.read_csv("cleaned_RET.csv", index_col='date')
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
Result = np.zeros(147)
Critical_left = np.zeros(147)
Critical_right = np.zeros(147)
average_ret = np.array(RET.mean())
average_ret_excess = np.array(RET_excess.mean())


# %% Apply to all factors
def nonlinear_beta_test(baseline_factor, FACTOR=FACTOR, RET_excess=RET_excess, with_intercept=1):
    j = 0
    count = 0
    for i in FACTOR.columns[~FACTOR.columns.isin(baseline_factor)]:
        tic = time.time()
        select = baseline_factor + [i]
        beta = np.array(OLSRegression(
            np.array(FACTOR[select]), RET_excess).beta_hat().iloc[:, 1:])
        Result[j], Critical_left[j], Critical_right[j] = my_bootstrap(
            beta, average_ret_excess, B=500, intercept=with_intercept)
        print("Factor", i, "Elapsed Time =", time.time() - tic)
        if Result[j] < Critical_left[j] or Result[j] > Critical_right[j]:
            print("The factor", i, "is nonlinear!")
            count = count+1
        j = j + 1
    print('Nonlinear Count =', count)
    return Result, Critical_left, Critical_right


# %% Different baseline factors.
# Individual
baseline_factor = []
[Result, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Result, Critical_left, Critical_right]
             ).to_csv('result_individual.csv')
# %% Market As baseline
baseline_factor = ['MktRf']
[Result, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)

# Result = {'Result': T_value,
#           '2.5% Quantile of Bootstrap Distribution': Critical_left,
#           '97.5% Quantile of Bootstrap Distribution': Critical_right
#           }
pd.DataFrame([Result, Critical_left, Critical_right]
             ).to_csv('result_baseline_mktrf.csv')
# %% FF3 as basline
baseline_factor = ['MktRf', 'HML', 'SMB']
[Result, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Result, Critical_left, Critical_right]
             ).to_csv('result_baseline_ff3.csv')

# %% Taming the Factor Zoo chooses 4 Factors:
# SMB, nxf, chcsho, pm
baseline_factor = ['SMB', 'nxf', 'chcsho', 'pm']
[Result, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Result, Critical_left, Critical_right]
             ).to_csv('result_baseline_TFZ4.csv')
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
j = 0
FNAME = list()
for i in FACTOR.columns:
    ff = i + ':::'
    FNAME.append(ff)
    print(ff)
FNAME = pd.DataFrame(FNAME)
FNAME.to_csv('FNAME.csv', index=False)


# %% Test on the Market Factor
beta = np.array(OLSRegression(
    np.array(FACTOR['MktRf']), RET_excess).beta_hat().iloc[:, 1:])
[a, b, c] = my_bootstrap(
    beta, average_ret_excess, B=500, intercept=1)
# %%
plt.figure()
plt.scatter(beta, average_ret_excess)
plt.scatter(beta, OLSRegression(beta, average_ret_excess).y_hat(intercept=0))
plt.scatter(beta, OLSRegression(beta, average_ret_excess).y_hat())
plt.scatter(beta, loc_poly(average_ret_excess, beta))
print(a, b, c)
#------<Some Additional Tests>------#
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
plt.xlim(0, 2)
plt.ylim(0, 0.02)
plt.scatter(beta[:, 0], m)
plt.scatter(beta[:, 0], m1)
plt.scatter(beta[:, 0], average_ret_excess)
plt.figure()
plt.xlim(-1, 1)
plt.ylim(0, 0.02)
plt.scatter(beta[:, 1], m)
plt.scatter(beta[:, 1], m1)
plt.scatter(beta[:, 1], average_ret_excess)
plt.figure()
plt.xlim(-1, 1)
plt.ylim(0, 0.02)
plt.scatter(beta[:, 2], m)
plt.scatter(beta[:, 2], m1)
plt.scatter(beta[:, 2], average_ret_excess)
a = my_bootstrap(beta, average_ret_excess)
b = my_bootstrap(beta, average_ret_excess, intercept=0)
print(a)
print(b)
