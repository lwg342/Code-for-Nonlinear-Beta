# -> Created on 22 October 2020
# -> Author: Weiguang Liu
#  %%
import pandas as pd
import numpy as np
# %%
DATA = pd.read_csv('cleaned_RET_EXCESS_port_202.csv', index_col = 0)
DATA.index = pd.to_datetime(DATA.index)
# %%
