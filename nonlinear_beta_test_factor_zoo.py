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
from tabulate import tabulate
DATA_Folder = '/Users/lwg342/OneDrive - University of Cambridge/Utility/Data/Stock data/FACTOR Zoo Data/data/'
# %% Apply to all factors


class NLBetaTest():
    def __init__(self, FACTOR, RET_EXCESS):
        from tabulate import tabulate
        self.FACTOR = FACTOR
        self.RET_EXCESS = RET_EXCESS
        self.Result = []
        self.with_intercept = 1
        self.bootstrap_iteration = 500
        self.estimating_period = 'full'
        self.testing_period = 'full'
        self.param_list = [
            ['Cross_sectional Regression has intercept', self.with_intercept],
            ['Bootstrap Iterations', self.bootstrap_iteration],
            ['Estimating Beta With', self.estimating_period],
            ['Estimating Average Excess Return with', self.testing_period],
        ]

    def describe(self):
        print('Header of FACTORs\n', self.FACTOR.iloc[0:3, 0:3], '\n')
        print('Header of Excess Return\n',
              self.RET_EXCESS.iloc[0:3, 0:3], '\n')
        print(tabulate(self.param_list, ['Parameter', 'Value']))

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
            average_excess_ret = np.array(
                self.RET_EXCESS[self.testing_period].mean())
        return average_excess_ret

    def test_model(self, baseline_factor, additional_factor=[], with_intercept=1):
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
                print("The factor model with",
                      self.model_factor, "is nonlinear!")
            full_name = '+'.join(baseline_factor + additional_factor)
            self.Result.append(
                [
                    full_name,
                    Tn,
                    Critical_left,
                    Critical_right,
                    Tn < Critical_left or Tn > Critical_right
                ])
        return

    def report(self):
        if self.Result == []:
            print('No result yet, fit the model first')
        else:
            print(tabulate(self.Result, [
                  'Model', 'Tn', 'Left CV', 'Right CV', 'Is Nonlinear?']))

    def save_result(self, name):
        address = 'result_' + name + '.csv'
        _df_ = pd.DataFrame(
            self.Result, columns=['Model', 'Tn', 'Left CV', 'Right CV', 'Is Nonlinear?']).set_index('Model')
        _df_.to_csv(address)
        with open('result' + name + '_document.txt', 'w') as _file_:
            for j in self.param_list:
                str_j = [str(i) for i in j]
                _file_.write(':'.join(str_j))
                _file_.write('\n')
            _file_.close()


# %% [markdown]
# # Import Data
# 1. Return data. Here I use S&P500 constituents, can be extended to more stocks
# 2. Import FACTOR data from FACTOR Zoo
# %% Import and cleaning data
# RET = pd.read_csv("cleaned_RET.csv", index_col='date')
# RET = pd.read_csv(
#     DATA_Folder + "port202.csv", index_col=0, header=None)
# RET.index = pd.to_datetime(RET.index, format='%Y%m')
# RET = RET/100 # This is special to this 202 return data
# RET.iloc[0:5, 0:5]

# %%
# %% 5*5 portfolio
RET = pd.read_csv(
    DATA_Folder + "port_5x5.csv", header=None)
RET = RET.drop([0], axis=1)
RET.index = pd.date_range("1976-07-31", "2017-12-31", freq='M')
RET.iloc[0:5, 0:5]
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


# %% Test
Model1 = NLBetaTest(FACTOR, RET_EXCESS)
Model1.describe()
Model1.test_model(['MktRf'])
Model1.test_model(['HML'])
Model1.test_model(['SMB'])
Model1.test_model(['nxf'])
Model1.test_model(['chcsho'])
Model1.test_model(['pm'])
Model1.test_model(['MktRf', 'HML', 'SMB'])
Model1.test_model(['SMB', 'nxf', 'chcsho', 'pm'])
Model1.report()
Model1.save_result('Models')
# %%
