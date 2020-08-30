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
DATA_Folder = '/Users/lwg342/OneDrive - University of Cambridge/Utility/Data/Stock data/FACTOR Zoo Data/data/'
# %% [markdown]
# # Import Data
# 1. Return data. Here I use S&P500 constituents, can be extended to more stocks
# 2. Import FACTOR data from FACTOR Zoo
# %% Import and cleaning data
# RET = pd.read_csv("cleaned_RET.csv", index_col='date')
RET = pd.read_csv(
    DATA_Folder + "port202.csv", index_col=0, header=None)
RET.index = pd.to_datetime(RET.index, format='%Y%m')
RET = RET/100
RET.iloc[0:5, 0:5]
# %% 5*5 portfolio
# RET = pd.read_csv(
#     DATA_Folder + "port_5x5.csv", header=None)
# RET = RET.drop([0], axis=1)
# RET.index = pd.date_range("1976-07-31", "2017-12-31", freq='M')
# RET.iloc[0:5, 0:5]
# %% Import FACTOR Data
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
RET_EXCESS = RET - np.array([FACTOR.RF]).T
FACTOR.RF.iloc[0:5]
RET_EXCESS.iloc[0:5, 0:5]
average_ret = np.array(RET.mean())
average_excess_ret = np.array(RET_EXCESS.mean())
# %% Apply to all factors
class NLBetaTest():
    def __init__(self, FACTOR, RET_EXCESS, Result = []):
        self.FACTOR = FACTOR
        self.RET_EXCESS = RET_EXCESS
        self.Result = Result
        self.with_intercept = 1
        self.bootstrap_iteration = 500
        self.estimating_period = 'full'
        self.testing_period = 'full'
    def report(self):
        print('Header of FACTORs\n', self.FACTOR.iloc[0:3,0:3], '\n')
        print('Header of Excess Return\n', self.RET_EXCESS.iloc[0:3,0:3], '\n')
        print('Result is', self.Result)
    def beta_estimate(self):
        # Estimating_period is a list of Booleans indicating which periods are used for estimating the betas
        if self.estimating_period == 'full':
            beta = np.array(OLSRegression(
                np.array(self.FACTOR[self.model_factor]), self.RET_EXCESS).beta_hat().iloc[:, 1:])
        else: 
            beta = np.array(OLSRegression(
                np.array(self.FACTOR.loc[self.estimating_period, self.model_factor]), self.RET_EXCESS).beta_hat().iloc[:, 1:])
        return beta
    def average_excess_return_estimate(self):
        if self.testing_period == 'full':
            average_excess_ret = np.array(self.RET_EXCESS.mean())
        else:
            average_excess_ret = np.array(self.RET_EXCESS[self.testing_period].mean())
        return average_excess_ret
    def test_model(self, baseline_factor, additional_factor = [], with_intercept=1):
        self.Result = []
        if any(i in baseline_factor for i in additional_factor):
            self.Result.append([additional_factor, '-----', '-----', '-----'])
            print('Additional FACTORs are in the Baseline Model')
        else:
            self.model_factor = baseline_factor + additional_factor
            beta = self.beta_estimate()
            average_excess_ret = self.average_excess_return_estimate()
            Tn, Critical_left, Critical_right = my_bootstrap(
                beta, average_excess_ret, B=self.bootstrap_iteration, intercept=self.with_intercept)
            if Tn < Critical_left or Tn > Critical_right:
                print("The factor model with", self.model_factor, "is nonlinear!")
            full_name = '+'.join(baseline_factor + additional_factor)
            self.Result.append([full_name, Tn, Critical_left, Critical_right])
        return 
    def save_result(self, name):
        address = '_'.join([name] + ['.csv'])
        pd.Dataframe(self.Result, index_col= 0).to_csv(address)
# def original_nonlinear_beta_test(baseline_factor, FACTOR=FACTOR, RET_EXCESS=RET_EXCESS, with_intercept=1):
#     Result = []
#     tic = time.time()
#     count = 0
#     for i in FACTOR.columns[0:5]:
#         if i in baseline_factor:
#             Result.append(
#                 [i, '-----', '-----', '-----'])
#         else:
#             select = baseline_factor + [i]
#             beta = np.array(OLSRegression(
#                 np.array(FACTOR[select]), RET_EXCESS).beta_hat().iloc[:, 1:])
#             Tn, Critical_left, Critical_right = my_bootstrap(
#                 beta, average_excess_ret, B=500, intercept=with_intercept)
#             Result.append([i, Tn, Critical_left, Critical_right])
#             if Tn < Critical_left or Tn > Critical_right:
#                 print("The factor", i, "is nonlinear!")
#                 count = count+1
#     print("Total Elapsed Time =", time.time() - tic)
#     print('Nonlinear Count =', count)
#     return Result
# %% Test
A = NLBetaTest(FACTOR, RET_EXCESS)
A.report()
A.one_model(['MktRf']).report()
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

# %% Taming the FACTOR Zoo chooses 4 FACTORs:
# SMB, nxf, chcsho, pm
baseline_factor = ['SMB', 'nxf', 'chcsho', 'pm']
[Tn, Critical_left, Critical_right] = nonlinear_beta_test(baseline_factor)
pd.DataFrame([Tn, Critical_left, Critical_right]
             ).to_csv('Result_baseline_TFZ4.csv')
# %% Taming the FACTOR Zoo chooses 4 FACTORs:
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
i = 3
plt.figure()
plt.scatter(beta[:, i], average_excess_ret)
plt.scatter(beta[:, i], OLSRegression(
    beta, average_excess_ret).y_hat(intercept=0))
plt.scatter(beta[:, i], OLSRegression(beta, average_excess_ret).y_hat())
# plt.scatter(beta[:,i], loc_poly(average_excess_ret, beta[:,i]))
print(a, b, c)

# %%
