# %% [markdown]
# In the next part we simulate the kernel-based test with sin() curve.
import numpy as np
from matplotlib import pyplot as plt
from myfunc import OLSRegression, kernel_test, my_bootstrap
# %% Simulation
np.random.seed(5)
X = np.arange(0, 5, 0.01)
N = X.shape[0]
# Y = np.sin(X) + np.random.normal(scale=1, size=N)
Y = 3 + 2*(X) + np.random.normal(scale=1, size=N)
m = OLSRegression(X, Y).y_hat(intercept= 1)
u = Y - m
plt.figure()
plt.scatter(X, Y)
plt.scatter(X, m)
plt.scatter(X, u)
Test_0 = kernel_test(u, X)
result = my_bootstrap(X, Y, B=100, intercept=1)

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

