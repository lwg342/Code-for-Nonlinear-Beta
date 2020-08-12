# %% import required libraries
import pandas as pd
import numpy as np
from scipy import linalg as LA
from scipy import stats as ST
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

# %% The local linear regression
# We take Gaussian Kernel
# Bandwidth is choosen as 1/T^0.4
# It can be used for multidimensional case. The plot is different
def loc_poly(Y, X):
    N, k = np.array([X]).T.shape
    m = np.empty(N)
    i = 0
    h = 1/(N**(2/5))
    # grid = np.arange(start=X.min(), stop=X.max(), step=np.ptp(X)/200)
    for x in X:
        Xx = np.mat(X).T - (np.ones([N, 1]))*x
        Xx1 = np.concatenate((np.ones([N, 1]),
                              Xx), axis=1)
        Wx = np.diag(ST.norm.pdf(Xx.T/h)[0])
        Sx = ((Xx1.T)@Wx@Xx1 + 1e-90*np.eye(k))
        m[i] = ((Sx.I) @ (Xx1.T) @ Wx @ np.mat(Y).T)[0]
        i = i + 1
    plt.figure()
    plt.scatter(X, Y)
    # plt.plot(grid, m, color= 'red')
    plt.scatter(X, m, color='red')
    return m

# %% Calculate the test statistic
def kernel_test(u, X):
    N = X.shape[0]
    inside_X = np.ones([N, N])
    if  X.ndim == 2:
        k = X.shape[1]
        for i in range(0, k):
            inside_X = inside_X * (np.array([X[:, i]]).T - X[:, i])
    else:
        k = 1
        inside_X = inside_X * (np.array([X]).T - X)
    h = 1/(N**(2/5))
    K = ST.norm.pdf((inside_X)/(h**k))
    K = K - np.diag(np.diag(K))
    I = u.T@K@u/N/(N-1)
    sigma_hat = np.sqrt(2*h/N/(N-1)*(u.T**2)@(K**2)@(u**2))
    T = N*(h**0.5)*I/sigma_hat
    return T, sigma_hat, k
# %% Return OLS estimate of the conditional mean as a col.vector
def OLS_mean(X, Y):
    m = sm.add_constant(X)@sm.OLS(Y, sm.add_constant(X)).fit().params.T
    return m


# %% Bootstrap Test

def my_bootstrap(X, Y, B=1000):
    N = Y.shape[0]
    m = OLS_mean(X, Y)
    u = Y - m
    Test_0 = kernel_test(u, X)[0]
    Test = np.empty(B)
    for j in range(0, B):
        # np.random.seed(j)
        u_star = ((0.5 + (-1)**np.random.binomial(n=1,
                                                  p=(1 + np.sqrt(5))/(2*np.sqrt(5)), size=N))*u)
        # plt.scatter(X,u)
        # plt.scatter(X,u_star)
        Y_star = m + u_star
        m_star = OLS_mean(X, Y_star)
        u_star_hat = Y_star - m_star
        Test[j] = kernel_test(u_star_hat, X)[0]
    Critical_left = np.quantile(Test, q=0.025)
    Critical_right = np.quantile(Test, q=0.975)

    # plt.figure()
    # plt.hist(Test, bins=30)
    # print(Test_0, Critical_left, Critical_right)
    return [Test_0, Critical_left, Critical_right]

# %%
