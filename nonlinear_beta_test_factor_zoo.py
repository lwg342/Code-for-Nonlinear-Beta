# %% Import Packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from myfunc import OLSRegression, kernel_test, loc_poly, my_bootstrap
import time
from tabulate import tabulate
import pickle # for saving and loading class instances
# %% Define NLBetaTest class


class NLBetaTest():
    def __init__(self, FACTOR, RET_EXCESS, name):
        self.FACTOR = FACTOR
        self.RET_EXCESS = RET_EXCESS
        self.name = name
        self.Result = []
        self.with_intercept = 1
        self.bootstrap_iteration = 1000
        self.estimating_period = FACTOR.index == FACTOR.index
        self.testing_period = FACTOR.index == FACTOR.index
        
    def _get_param_list(self):
        param_list = [
            ['Cross Sectional Regression has intercept', self.with_intercept],
            ['Bootstrap Iterations', self.bootstrap_iteration],
            ['Estimating Beta With', str(
                self.FACTOR.index[self.estimating_period][0].date()) + ' to ' + str(self.FACTOR.index[self.estimating_period][-1].date())],
            ['Estimating Average Excess Return with', str(
                self.FACTOR.index[self.testing_period][0].date()) +'-'+ str(self.FACTOR.index[self.testing_period][-1].date())],
            ['Return Dimension TxN', self.RET_EXCESS.shape],
        ]
        return param_list
    def _result_to_df(self):
        _df = pd.DataFrame(
            self.Result, columns=['Model', 'Tn', 'Left CV', 'Right CV', 'Is Nonlinear?']).set_index('Model')
        return _df
    def describe(self):
        print('Header of FACTORs\n', self.FACTOR.iloc[0:3, 0:3], '\n')
        print('Header of Excess Return\n',
              self.RET_EXCESS.iloc[0:3, 0:3], '\n')
        print(tabulate(self._get_param_list(), ['Parameter', 'Value']))
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
            self.Result.append([additional_factor, Nan, Nan, Nan])
            print('Additional FACTORs are in the Baseline Model')
        else:
            self.model_factor = baseline_factor + additional_factor
            beta = self.beta_estimate()
            average_excess_ret = self.average_excess_return_estimate()
            Tn, critical_left, critical_right = my_bootstrap(
                beta, average_excess_ret, B=self.bootstrap_iteration, intercept=self.with_intercept)
            if Tn < critical_left or Tn > critical_right:
                print("The factor model with",
                      self.model_factor, "is nonlinear!")
            full_name = '+'.join(self.model_factor)
            self.Result.append(
                [
                    full_name,
                    Tn,
                    critical_left,
                    critical_right,
                    Tn < critical_left or Tn > critical_right
                ])
        return

    def report(self):
        if self.Result == []:
            print('No result yet, fit the model first')
        else:
            print(tabulate(self._result_to_df()))

    def save_to_markdown(self, folder = ''):
        with open(folder + 'result_' + self.name + '.md', 'w') as _file:
            _file.write('---\n'+'title: '+ self.name + '\n---\n\n')
            _file.write('# '+ self.name + '\n\n')
            for j in self._get_param_list():
                _file.write('- ')
                str_j = [str(i) for i in j]
                _file.write(':\t'.join(str_j))
                _file.write('\n')
            _file.write('\n\n\n')
            _file.write(self._result_to_df().round(2).replace(False,'').replace(True, '*').to_markdown())

    def plot(self):
        i = 0
        for j in self.model_factor:
            plt.figure()
            plt.scatter(self.beta_estimate()[
                        :, i], self.average_excess_return_estimate())
            plt.scatter(self.beta_estimate()[:, i], OLSRegression(X=self.beta_estimate(
            ), Y=self.average_excess_return_estimate()).y_hat(self.with_intercept))
            i = i+1
def publish_result(models):
    for j in models:
        j.save_to_markdown(
        folder='/Users/lwg342/Documents/GitHub/Site-Generation/content/Nonlinear Beta Tests Results/')
    return 1
# %%
with open('save.dat','rb') as _file:
    models = pickle.load(_file)


# %%
with open('save.dat', 'wb') as _file:
    pickle.dump(models, _file)
