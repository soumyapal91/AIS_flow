import numpy as np
# import torch

from utils import *
from utils_torch import *
# from scipy.linalg import lu, cholesky, solve, solve_triangular, norm, sqrtm
from scipy.stats import multivariate_normal


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



dim = 2
nsample = 1

x = np.random.rand(nsample, dim)

mu = np.zeros(dim)
cov_sqrt = np.random.rand(dim, dim)
cov = np.dot(cov_sqrt, cov_sqrt.T) + np.eye(dim)
cov = np.tile(cov[:, :, None], [1, 1, nsample])

print('-------------')
print(loggausspdf(x, mu, cov[:, :, 0]))
print('-------------')
print(loggausspdf(x, mu, cov))
print('-------------')
var = multivariate_normal(mean=mu, cov=cov[:, :, 0])
print(np.log(var.pdf(x)))
print('-------------')
print(loggausspdf_t(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(cov)))

#
#
# print(cov[:, :, 0])
# print(scipy.linalg.cholesky(cov[:, :, 0], lower=True))
#
#
# x = np.random.rand(5, dim)
# mu = np.random.rand(3, dim)
# Sigma = np.tile(cov[:, :, [0]], [1, 1, 3])
#
# print(logGMMpdf(x, mu, Sigma))
# print(logGMMpdf_t(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(Sigma)))
#
# print(cov.shape)
#
# for k in range(cov.shape[-1]):
#     print(scipy.linalg.cholesky(cov[:, :, k], lower=True))
#     print('duck')
#     print(torch.linalg.cholesky(torch.from_numpy(cov[:, :, k])))
#     print('tuck')
#     print('-----------------------')
