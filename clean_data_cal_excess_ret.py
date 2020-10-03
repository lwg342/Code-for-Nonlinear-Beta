# %%
data_path = 'original_data/'
# %% Import FACTOR Data
FACTOR = pd.read_csv(data_path + 'factors_zoo.csv')
FACTOR.rename(columns={'  Date': 'date'}, inplace=True)
FACTOR.date = pd.to_datetime(FACTOR.date, format='%Y%m%d')
FACTOR = FACTOR.pivot_table(index='date')
FACTOR = FACTOR.dropna(axis=1)
FACTOR.iloc[0:5, 0:5]
FACTOR.to_csv('cleaned_factor_zoo.csv')

# %% Clean Holding Period Return Data
DATA = pd.read_csv(
    data_path +'HPRET_monthly_correspondint_to_factor_zoo_all.csv')
DATA.date = pd.to_datetime(DATA.date, format='%Y%m%d')
DATA = DATA.drop(DATA[DATA.RET == 'C'].index)
DATA = DATA.drop(DATA[DATA.RET == 'B'].index)
DATA.RET = DATA['RET'].astype('float')
RET = DATA.pivot_table('RET', index='PERMNO', columns='date')
RET = RET.dropna(0).transpose()
RET_EXCESS = RET - np.array([FACTOR.RF]).T
RET_EXCESS.to_csv("cleaned_RET_EXCESS_HPRET.csv")

# %% Portfolio 5x5
RET = pd.read_csv(data_path + "port_5x5.csv", header=None)
RET = RET.drop([0], axis=1)
RET.index = pd.date_range("1976-07-31", "2017-12-31", freq='M')
RET.iloc[0:5, 0:5]
RET_EXCESS = RET - np.array([FACTOR.RF]).T
RET_EXCESS.to_csv("cleaned_RET_EXCESS_port_5x5.csv")

# %% Portfolio 202
RET = pd.read_csv(data_path + "port202.csv", index_col=0, header=None)
RET.index = pd.to_datetime(RET.index, format='%Y%m')
RET = RET/100  # This is special to this 202 return data
RET.iloc[0:5, 0:5]
RET_EXCESS = RET - np.array([FACTOR.RF]).T
RET_EXCESS.to_csv("cleaned_RET_EXCESS_port_202.csv")

# %%
