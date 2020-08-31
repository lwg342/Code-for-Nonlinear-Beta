# %% [markdown]
# In the next part we simulate the kernel-based test with sin() curve.
import numpy as np
from matplotlib import pyplot as plt
from myfunc import OLSRegression, kernel_test, my_bootstrap
# %% Simulation
np.random.seed(10)
sigma = 1
q = 1 # number of covariates
X = np.arange(0, 1, 0.0001)
N = X.shape[0]
# Y = np.sin(X) + sigma* np.random.normal(scale=1, size=N)
Y = 3 + 2*(X**2) + np.random.normal(scale=1, size=N)
m = OLSRegression(X, Y).y_hat(intercept= 1)
u = Y - m
plt.figure()
plt.scatter(X, Y)
plt.scatter(X, m)
plt.scatter(X, u)
Test_0 = kernel_test(u, X)
print(N,Test_0)
kappa = 1/(2*sigma*np.sqrt(np.pi))
theoretical_sigma_Tn = np.sqrt(2*(kappa**q)*3*((1/N)**q))
print(theoretical_sigma_Tn)
# %%
result = my_bootstrap(X, Y, B=500, intercept=1)
print(result)
# %% A simulation of multivariate relationship
np.random.seed(6)
X = np.array([np.arange(0, 5, 0.01),np.random.normal(size= 500)]).T
N = X.shape[0]
beta = np.array([1,2])
Y = np.sin(X)@beta + 0.5* X[:,0]* np.random.normal(scale=1, size=N)
m = OLSRegression(X, Y).y_hat()
u = Y - m
plt.figure()
plt.scatter(X[:,0], Y)
plt.scatter(X[:,1], Y)
# plt.scatter(X, m)
# plt.scatter(X, u)
Test_0 = kernel_test(u, X)
result = my_bootstrap(X, Y, B=500)


# %%
