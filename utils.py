import numpy as np
from scipy.linalg import cholesky, solve_triangular


def isdiag(P):
    P = np.squeeze(P)
    return np.count_nonzero(P - np.diag(np.diag(P))) == 0


def logdet(A):
    assert isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[0] == A.shape[1], \
        'A should be a square matrix of double or single class.'
    return 2 * np.sum(np.log(np.diag(cholesky(A, lower=True))))


def loggausspdf(xp, x0, P0):
    # Calculate log of d-dimensional Gaussian prob. density evaluated at xp
    # with mean x0 and covariance P0
    #
    # xp: particles: dimensions N x d
    # x0: mean: 1 by d
    # P0: covariance: d by d (by N)
    #
    # g: evaluation of the log-pdf
    # chisq: the (xp-x)'inv(P0)(xp-x)
    if xp.ndim == 1:
        xp = xp[None, :]

    N, d = xp.shape

    twopi_factor = 0.5 * d * np.log(2 * np.pi)

    if x0.ndim == 1:
        y = xp - np.tile(x0, (N, 1))
    else:
        y = xp - x0

    if P0.ndim == 3 and np.all(np.diff(P0, axis=-1)) == 0.0:
        P0 = P0[:, :, 0]

    if P0.ndim == 2:
        if isdiag(P0):
            chisq = np.sum((y ** 2) / np.tile(np.diag(P0).reshape(1, -1), (N, 1)), axis=1)
        else:
            L = cholesky(P0, lower=True)
            alpha = np.array([solve_triangular(L, y[i], lower=True) for i in range(N)])
            chisq = np.sum(alpha * alpha, axis=1)
        g = -(chisq / 2) - twopi_factor - 0.5 * logdet(P0)
    else:
        g = np.zeros(N)
        for i in range(N):
            if isdiag(P0[:, :, i]):
                chisq = np.sum((y[i] ** 2) / np.diag(P0[:, :, i]))
            else:
                L = cholesky(P0[:, :, i], lower=True)
                alpha = solve_triangular(L, y[i], lower=True)
                chisq = np.sum(alpha * alpha)
            g[i] = -(chisq / 2) - twopi_factor - 0.5 * logdet(P0[:, :, i])

    return np.squeeze(g)


def logGMMpdf(xp, mu, Sigma, alpha=None):
    # Compute the log-pdf of a Gaussian Mixture Model (GMM) at given particles

    # xp : N x d, each column is a particle
    # mu : k x d, each column is a mean vector in GMM
    # Sigma : d x d x k, covariance matrices of GMM components
    # alpha : k, component proportions in GMM, uniform if argument not specified
    # d : dimension of state space
    # k : number of Gaussian components in GMM
    # n : number of particles
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

    logpdf = max_term + np.log(np.sum(np.exp(logScaledComp - np.tile(max_term, (k, 1))), axis=0))

    return np.squeeze(logpdf)


def resample(n, logW):
    # resample n particles from a weighted set with log weights logW

    logW = logW.reshape(1, -1)
    w = np.exp(logW - np.max(logW))
    w = w / np.sum(w)

    edges = np.cumsum(w)
    edges[-1] = 1.0  # get the upper edge exact
    return np.squeeze(np.searchsorted(edges, np.random.rand(n)))


def particle_estimate(particles, log_weights, normalizing_constant=False):
    # Form estimate based on weighted set of particles

    # log_weights: logarithmic weights [1xN]
    # particles: the state values of the particles [Nxdim]

    if normalizing_constant is None:
        normalizing_constant = False

    particle_inf_nan = np.logical_or(np.isnan(particles), np.isinf(particles)).any(axis=1)
    particles = particles[~particle_inf_nan, :]
    log_weights = log_weights[~particle_inf_nan]

    if normalizing_constant:
        estimate = np.mean(np.exp(log_weights))
    else:
        N, dim = particles.shape
        weights = np.exp(log_weights - np.max(log_weights))
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
        weights /= np.sum(weights)
        weights_repeated = np.tile(weights.reshape(-1, 1), (1, dim))
        estimate = np.sum(particles * weights_repeated, axis=0)

    return estimate


def MSE(x_est, x_true):
    x_est = np.squeeze(x_est)
    x_true = np.squeeze(x_true)
    return np.mean((x_est - x_true) ** 2)



