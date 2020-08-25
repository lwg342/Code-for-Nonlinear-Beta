DATA = pd.read_csv(
    'HPRET_monthly_correspondint_to_factor_zoo_all.csv')
DATA.date = pd.to_datetime(DATA.date, format='%Y%m%d')

DATA = DATA.drop(DATA[DATA.RET == 'C'].index)
DATA = DATA.drop(DATA[DATA.RET == 'B'].index)
DATA.RET = DATA['RET'].astype('float')
RET = DATA.pivot_table('RET', index='PERMNO', columns='date')
RET = RET.dropna(0).transpose()
RET.to_csv("cleaned_RET.csv")