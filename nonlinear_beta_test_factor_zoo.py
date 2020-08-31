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
# %% Apply to all factors


class NLBetaTest():
    def __init__(self, FACTOR, RET_EXCESS):
        self.FACTOR = FACTOR
        self.RET_EXCESS = RET_EXCESS
        self.Result = []
        self.with_intercept = 1
        self.bootstrap_iteration = 1000
        self.estimating_period = FACTOR.index == FACTOR.index
        self.testing_period = FACTOR.index == FACTOR.index
        
    def _get_param_list_(self):
        param_list = [
            ['Cross_sectional Regression has intercept', self.with_intercept],
            ['Bootstrap Iterations', self.bootstrap_iteration],
            ['Estimating Beta With', str(
                self.FACTOR.index[self.estimating_period][0]) + '-' + str(self.FACTOR.index[self.estimating_period][-1])],
            ['Estimating Average Excess Return with', str(
                self.FACTOR.index[self.testing_period][0]) +'-'+ str(self.FACTOR.index[self.testing_period][-1])],
            ['Return Dimension', self.RET_EXCESS.shape],
        ]
        return param_list

    def describe(self):
        print('Header of FACTORs\n', self.FACTOR.iloc[0:3, 0:3], '\n')
        print('Header of Excess Return\n',
              self.RET_EXCESS.iloc[0:3, 0:3], '\n')
        print(tabulate(self._get_param_list_(), ['Parameter', 'Value']))
        print('\n')

    def beta_estimate(self):
        # Estimating_period is a list of Booleans indicating which periods are used for estimating the betas
        if self.estimating_period == 'full':
            beta = np.array(OLSRegression(
                np.array(self.FACTOR[self.model_factor]), self.RET_EXCESS).beta_hat().iloc[:, 1:])
        else:
            beta = np.array(OLSRegression(
                np.array(self.FACTOR.loc[self.estimating_period, self.model_factor]), self.RET_EXCESS.loc[self.estimating_period]).beta_hat().iloc[:, 1:])
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
        with open('result_' + name + '_document.txt', 'w') as _file_:
            for j in self._get_param_list_():
                str_j = [str(i) for i in j]
                _file_.write(':'.join(str_j))
                _file_.write('\n')
            _file_.close()

    def plot(self):
        i = 0
        for j in self.model_factor:
            plt.figure()
            plt.scatter(self.beta_estimate()[
                        :, i], self.average_excess_return_estimate())
            plt.scatter(self.beta_estimate()[:, i], OLSRegression(X=self.beta_estimate(
            ), Y=self.average_excess_return_estimate()).y_hat(self.with_intercept))
            i = i+1
# %% [markdown]
# # Import Data
# 1. Return data. Here I use S&P500 constituents, can be extended to more stocks
# 2. Import FACTOR data from FACTOR Zoo
# %% Import FACTOR Data
FACTOR = pd.read_csv('factors_zoo.csv')
FACTOR.rename(columns={'  Date': 'date'}, inplace=True)
FACTOR.date = pd.to_datetime(FACTOR.date, format='%Y%m%d')
FACTOR = FACTOR.pivot_table(index='date')
FACTOR = FACTOR.dropna(axis=1)
FACTOR.iloc[0:5, 0:5]
# %% [markdown]
# # Individual test of nonlinearity
# %%
RET = pd.read_csv("cleaned_RET.csv", index_col='date')
RET_EXCESS = RET - np.array([FACTOR.RF]).T
# %% Test
m1 = NLBetaTest(FACTOR, pd.concat([RET_EXCESS,RET_EXCESS],axis=1))
m1 = NLBetaTest(FACTOR, RET_EXCESS)
m1.describe()
m1.test_model(['MktRf'])
m1.test_model(['HML'])
m1.test_model(['SMB'])
m1.test_model(['nxf'])
m1.test_model(['chcsho'])
m1.test_model(['pm'])
m1.test_model(['MktRf', 'HML', 'SMB'])
m1.test_model(['SMB', 'nxf', 'chcsho', 'pm'])
m1.report()
# m1.plot()
m1.save_result('hpret')
# %%
m6 = NLBetaTest(FACTOR, RET_EXCESS)
m6.estimating_period = FACTOR.index < '2000-01-01'
m6.testing_period = FACTOR.index >= '2000-01-01'
m6.describe()
m6.test_model(['MktRf'])
m6.test_model(['HML'])
m6.test_model(['SMB'])
m6.test_model(['nxf'])
m6.test_model(['chcsho'])
m6.test_model(['pm'])
m6.test_model(['MktRf', 'HML', 'SMB'])
m6.test_model(['SMB', 'nxf', 'chcsho', 'pm'])
m6.report()
# m6.plot()
m6.save_result('hpret_using_subsample')

# %% 5*5 portfolio
RET = pd.read_csv("port_5x5.csv", header=None)
RET = RET.drop([0], axis=1)
RET.index = pd.date_range("1976-07-31", "2017-12-31", freq='M')
RET.iloc[0:5, 0:5]
RET_EXCESS = RET - np.array([FACTOR.RF]).T

# %%
m2 = NLBetaTest(FACTOR, RET_EXCESS)
m2.describe()
m2.test_model(['MktRf'])
m2.test_model(['HML'])
m2.test_model(['SMB'])
m2.test_model(['nxf'])
m2.test_model(['chcsho'])
m2.test_model(['pm'])
m2.test_model(['MktRf', 'HML', 'SMB'])
m2.test_model(['SMB', 'nxf', 'chcsho', 'pm'])
m2.report()
# m2.plot()
m2.save_result('Models_port_5x5')
# %%
m5 = NLBetaTest(FACTOR, RET_EXCESS)
m5.estimating_period = FACTOR.index < '2000-01-01'
m5.testing_period = FACTOR.index >= '2000-01-01'
m5.describe()
m5.test_model(['MktRf'])
m5.test_model(['HML'])
m5.test_model(['SMB'])
m5.test_model(['nxf'])
m5.test_model(['chcsho'])
m5.test_model(['pm'])
m5.test_model(['MktRf', 'HML', 'SMB'])
m5.test_model(['SMB', 'nxf', 'chcsho', 'pm'])
m5.report()
# m5.plot()
m5.save_result('Models_port_5x5_using_subsample')
# %%
RET = pd.read_csv("port202.csv", index_col=0, header=None)
RET.index = pd.to_datetime(RET.index, format='%Y%m')
RET = RET/100  # This is special to this 202 return data
RET.iloc[0:5, 0:5]
RET_EXCESS = RET - np.array([FACTOR.RF]).T
# %%
m3 = NLBetaTest(FACTOR, RET_EXCESS)
m3.describe()
m3.test_model(['MktRf'])
m3.test_model(['HML'])
m3.test_model(['SMB'])
m3.test_model(['nxf'])
m3.test_model(['chcsho'])
m3.test_model(['pm'])
m3.test_model(['MktRf', 'HML', 'SMB'])
m3.test_model(['SMB', 'nxf', 'chcsho', 'pm'])
m3.report()
# m3.plot()
m3.save_result('Models_port_202')

# %%
m4 = NLBetaTest(FACTOR, RET_EXCESS)
m4.estimating_period = FACTOR.index < '2000-01-01'
m4.testing_period = FACTOR.index >= '2000-01-01'
m4.describe()
m4.test_model(['MktRf'])
m4.test_model(['HML'])
m4.test_model(['SMB'])
m4.test_model(['nxf'])
m4.test_model(['chcsho'])
m4.test_model(['pm'])
m4.test_model(['MktRf', 'HML', 'SMB'])
m4.test_model(['SMB', 'nxf', 'chcsho', 'pm'])
m4.report()
# m4.plot()
m4.save_result('Models_port_202_using_subsample')

# %%
# Iteration
m7 = NLBetaTest(FACTOR,RET_EXCESS)
for i in FACTOR.columns:
    m7.test_model(baseline_factor = [], additional_factor= [i])
m7.report()
# %%
