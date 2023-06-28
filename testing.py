import numpy as np
# import torch

from utils import *
from utils_torch import *
from scipy.linalg import lu, cholesky, solve, solve_triangular, norm, sqrtm
from scipy.stats import multivariate_normal

import torch, time


def is_pos_def(A):
    if isdiag(A):
        if np.all(np.diag(A) > 0):
            return True
        else:
            return False
    elif np.array_equal(A, A.T):
        try:
            cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def cov_regularize(cova):
    # Regularize covariance matrices if the Cholesky factorization is not positive definite
    dim = cova.shape[0]
    reg = np.eye(dim) * 1e-14

    # Perform Cholesky decomposition
    indicator = is_pos_def(cova)
    count = 0
    maxCount = 100

    # Check whether the factorization is positive definite
    # If not, add regularization matrix to the covariance matrix
    while (not indicator) and count < maxCount:
        cova += reg
        indicator = is_pos_def(cova)
        count += 1

    # throw an exception if positive-definiteness cannot be achieved
    if count == maxCount:
        raise Exception('cov_regularize:TooManyIterations', 'Could not regularize the covariance matrix')

    return cova


def is_pos_def_t(A):
    if isdiag_t(A):
        if torch.all(torch.diag(A) > 0):
            return True
        else:
            return False
    elif torch.equal(A, A.T):
        try:
            cholesky(A)
            return True
        except:
            return False
    else:
        return False


def cov_regularize_t(cova):
    # Regularize covariance matrices if the Cholesky factorization is not positive definite
    dim = cova.shape[0]
    reg = torch.eye(dim) * 1e-14

    # Perform Cholesky decomposition
    indicator = is_pos_def_t(cova)
    count = 0
    maxCount = 100

    # Check whether the factorization is positive definite
    # If not, add regularization matrix to the covariance matrix
    while not indicator and count < maxCount:
        cova += reg
        indicator = is_pos_def_t(cova)
        count += 1

    # Throw an exception if positive-definiteness cannot be achieved
    if count == maxCount:
        raise Exception('cov_regularize:TooManyIterations', 'Could not regularize the covariance matrix')
        # or print a warning message
        print('cov_regularize:TooManyIterations', 'Could not regularize the covariance matrix')

    return cova


def posterior_latent_GMM(xp, mu, Sigma, alpha=None):
    # Compute the posterior of the latent variable in a Gaussian Mixture Model (GMM) at given particles

    # xp : N x d, each column is a particle
    # mu : k x d, each column is a mean vector in GMM
    # Sigma : d x d x k, covariance matrices of GMM components
    # alpha : k, component proportions in GMM, uniform if argument not specified
    # d : dimension of state space
    # k : number of Gaussian components in GMM
    # n : number of particles
    # posterior of latent: n x k, each row sums up to 1

    if xp.ndim == 1:
        xp = xp[None, :]

    N = xp.shape[0]
    k = mu.shape[0]

    if alpha is None:
        alpha = np.ones(k)

    alpha = alpha / np.sum(alpha)

    logScaledComp = np.zeros((k, N))

    for i in range(k):
        logScaledComp[i, :] = loggausspdf(xp, mu[i], Sigma[:, :, i]) + np.log(alpha[i])

    max_term = np.max(logScaledComp, axis=0)

    logGMMpdf = max_term + np.log(np.sum(np.exp(logScaledComp - np.tile(max_term, (k, 1))), axis=0))

    term = logScaledComp - np.tile(logGMMpdf, (k, 1))
    posterior_latent = np.exp(term - np.max(term)).T
    posterior_latent = posterior_latent / np.sum(posterior_latent, axis=1)[:, None]

    return np.squeeze(posterior_latent)


dim = 50
nsample = 1000

x = np.random.rand(nsample, dim)

mu = np.zeros(dim)
cov_sqrt = np.random.rand(dim, dim)
cov = np.dot(cov_sqrt, cov_sqrt.T) + np.eye(dim)
cov = np.tile(cov[:, :, None], [1, 1, nsample])

# print('-------------')
# print(loggausspdf(x, mu, cov[:, :, 0]))
# print('-------------')
# print(loggausspdf(x, mu, cov))
# print('-------------')
# var = multivariate_normal(mean=mu, cov=cov[:, :, 0])
# print(np.log(var.pdf(x)))
# print('-------------')
# print(loggausspdf_t(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(cov)))
# print('-------------')
# print(log_multivariate_gaussian(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(cov[:, :, 0])))

print('-------------')
x = torch.from_numpy(x)
mu = torch.from_numpy(mu)
cov = torch.from_numpy(cov)
start = time.time()

for i in range(5):
    z1 = loggausspdf_t(x, mu, cov)
    # z2 = log_multivariate_gaussian(x, mu, cov[:, :, 0])
    # print(torch.sum(torch.abs(z1-z2)).item())

print(time.time()-start)


print('-------------')
start = time.time()

for i in range(5000):
    # z1 = np.linalg.inv(np.eye(2))
    z2 = np.linalg.solve(np.eye(2), np.eye(2))
    # print(torch.sum(torch.abs(z1-z2)).item())

print(time.time()-start)