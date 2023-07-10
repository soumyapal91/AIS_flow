import numpy as np
# import torch

from utils import *
from utils_torch import *
from scipy.linalg import lu, cholesky, solve, solve_triangular, norm, sqrtm
from scipy.stats import multivariate_normal

import torch, time


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
#
# import cProfile
# from scipy.stats import invwishart
#
# cov = invwishart.rvs(df=200, scale=np.eye(200), size=1) + 2 * np.eye(200)
#
# start = time.time()
#
# for i in range(1000):
#     L = torch.linalg.inv(torch.from_numpy(cov))
#
# print(time.time()-start)
#
# dim = 10
# nsample = 12
# x = np.random.rand(nsample, dim)
# mu = np.random.rand(dim)
# cov = invwishart.rvs(df=dim, scale=np.eye(dim), size=1) + 2 * np.eye(dim)
#
# print('----------------------------')
# print(loggausspdf(x, mu, cov))
# print(loggausspdf_target(x, mu, cov, np.linalg.inv(cov), logdet(cov)))
# print(loggausspdf_t(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(cov)))
# print(loggausspdf_target_t(torch.from_numpy(x), torch.from_numpy(mu), torch.from_numpy(cov),  torch.linalg.inv(torch.from_numpy(cov)), logdet_t(torch.from_numpy(cov))))
#
# yo = logdet(cov)
#
# # print(logdet_t(torch.from_numpy(cov)))
# print(torch.tensor(yo))

