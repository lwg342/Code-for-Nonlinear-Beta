# %% Import Libraries
import numpy as np 
import pandas as pd
from myfunc import kernel_test, my_bootstrap, OLSRegression, nonlinear_beta_test

# %% [markdown] 
# ## Data 
# The data is from CRSP beta-suite, consisting of daily data of 253 stocks, all of which are constituents of SP500, from 1990-01-02 to 2019-12-31. The factors are FF 3 factors plus Carhart's factor. 


# %% import dataset
T = pd.read_csv(
    "/Users/lwg342/Dropbox/Cam Econ Trio/Data/cleaned_sp500_1990_2019_4factors_beta_suite.csv", index_col='PERMNO')
print(T.head())
T.columns
T.DATE = pd.to_datetime(T.DATE)
date = T['DATE'].unique()
name = T.index.unique()

# %% Import factor data 
Factor = pd.read_csv("F-F_Research_Data_Factors_daily.CSV",index_col = 'Date')
Factor.index = pd.to_datetime(Factor.index,format = '%Y%m%d')
RF = Factor.loc[date, 'RF']
RF.iloc[:5]
# %% [markdown] 
# In the next cell we create the matrices for returns, excess returns and idiosyncratic volatility. 
# %%  Create the matrices    
ret= pd.pivot_table(T, 'RET', index = 'PERMNO',columns = 'DATE')
ret.iloc[:5,:5]
RET_excess = np.transpose(ret - RF)
RET_excess.iloc[:5,:5]
Result = np.zeros(147)
Critical_left = np.zeros(147)
Critical_right = np.zeros(147)
average_ret_excess = np.array(RET_excess.mean())

# %%
result = np.empty(3)
Result = my_bootstrap()
# %%
