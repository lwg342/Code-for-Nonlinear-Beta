
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

# %%  For printing the names
# %%
j = 0
FNAME = list()
for i in FACTOR.columns:
    ff = i + ':::'
    FNAME.append(ff)
    print(ff)
FNAME = pd.DataFrame(FNAME)
FNAME.to_csv('FNAME.csv', index=False)

# %%# For testing a specific model
# %% Test on a specific model
beta = OLSRegression(
    np.array(FACTOR[['SMB', 'nxf', 'chcsho', 'pm']]), np.array(RET)).beta_hat()[:, 1:]
second_step_OLS = sm.OLS(sm.add_constant(beta), average_ret_excess).fit()
second_step_OLS.params
[a, b, c] = my_bootstrap(
    beta, average_ret_excess, B=500, intercept=1)

# %% 2020-08-31

# %% Test
m1 = NLBetaTest(FACTOR, pd.concat([RET_EXCESS, RET_EXCESS], axis=1))
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
m7 = NLBetaTest(FACTOR, RET_EXCESS)
for i in FACTOR.columns:
    m7.test_model(baseline_factor=[], additional_factor=[i])
m7.report()
# %%
FACTOR = pd.read_csv('cleaned_factor_zoo.csv', index_col='date')
RET_EXCESS = pd.read_csv('cleaned_RET_EXCESS_port_202.csv', index_col=0)
m7 = NLBetaTest(FACTOR, RET_EXCESS,
                name='Portfolio 202 adding one factor to Feng et al 4 Factors')
for j in FACTOR.columns:
    m.test_model(baseline_factor=['SMB', 'nxf',
                                  'chcsho', 'pm'], additional_factor=[j])
m.report()
models.append(m)

# %%
FACTOR = pd.read_csv('cleaned_factor_zoo.csv', index_col='date')
FACTOR.index = pd.to_datetime(FACTOR.index)
RET_EXCESS = pd.read_csv('cleaned_RET_EXCESS_HPRET.csv', index_col=0)
# %%
m8 = NLBetaTest(FACTOR, RET_EXCESS,
                name='Portfolio Holding Period adding one factor to Feng et al 4 Factors')
for j in FACTOR.columns:
    m.test_model(baseline_factor=['SMB', 'nxf',
                                  'chcsho', 'pm'], additional_factor=[j])
m.report()
models.append(m8)
# %%
k1 = 0
k2 = 0
for i in models[7].Result:
    for j in i:
        if j == '-----':
            models[7].Result[k1][k2] = np.nan
        k2 = k2 + 1
    k1 = k1 + 1
    k2 = 0
k1 = 0
k2 = 0
for i in models[8].Result:
    for j in i:
        if j == '-----':
            models[8].Result[k1][k2] = np.nan
        k2 = k2 + 1
    k1 = k1 + 1
    k2 = 0
