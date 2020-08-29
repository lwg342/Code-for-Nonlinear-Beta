
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
