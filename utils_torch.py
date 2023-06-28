import numpy as np
import torch
import torch.nn as nn
from torch.linalg import cholesky
from flows import *
from utils_torch import *
import torch.distributions as dist


def isdiag_t(P):
    P = torch.squeeze(P)
    return torch.count_nonzero(P - torch.diag(torch.diag(P))) == 0


def logdet_t(A):
    assert isinstance(A, torch.Tensor) and A.ndim == 2 and A.shape[0] == A.shape[1], \
        'A should be a square matrix of double or single class.'
    L = torch.linalg.cholesky(A)
    return 2 * torch.sum(torch.log(torch.diag(L)))


def loggausspdf_t(xp, x0, P0):
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
        xp = xp.unsqueeze(0)

    N, d = xp.shape

    twopi_factor = torch.tensor(0.5 * d * np.log(2 * np.pi))

    if x0.ndim == 1:
        y = xp - x0.repeat(N, 1)
    else:
        y = xp - x0

    if P0.ndim == 3 and torch.all(torch.diff(P0, dim=-1)) == 0.0:
        P0 = P0[:, :, 0]

    if P0.ndim == 2:
        if isdiag_t(P0):
            chisq = torch.sum((y ** 2) / torch.tile(torch.diag(P0).reshape(1, -1), (N, 1)), dim=1)
        else:
            L = cholesky(P0)
            alpha = torch.stack([torch.triangular_solve(y[i].unsqueeze(0).T, L, upper=False).solution.squeeze() for i in range(N)])
            chisq = torch.sum(alpha * alpha, dim=1)
        g = -(chisq / 2) - twopi_factor - 0.5 * logdet_t(P0)
    else:
        g = torch.zeros(N)
        for i in range(N):
            if isdiag_t(P0[:, :, i]):
                chisq = torch.sum((y[i] ** 2) / torch.diag(P0[:, :, i]))
            else:
                L = cholesky(P0[:, :, i])
                alpha = torch.triangular_solve(y[i].unsqueeze(0).T, L, upper=False).solution.squeeze()
                chisq = torch.sum(alpha * alpha)
            g[i] = -(chisq / 2) - twopi_factor - 0.5 * logdet_t(P0[:, :, i])

    return g.squeeze()


def logGMMpdf_t(xp, mu, Sigma, alpha=None):
    # Compute the log-pdf of a Gaussian Mixture Model (GMM) at given particles

    # xp : N x d, each column is a particle
    # mu : k x d, each column is a mean vector in GMM
    # Sigma : d x d x k, covariance matrices of GMM components
    # alpha : k, component proportions in GMM, uniform if argument not specified
    # d : dimension of state space
    # k : number of Gaussian components in GMM
    # n : number of particles
    if xp.ndim == 1:
        xp = xp.unsqueeze(0)

    N = xp.shape[0]
    k = mu.shape[0]

    if alpha is None:
        alpha = torch.ones(k)

    alpha = alpha / torch.sum(alpha)

    logScaledComp = torch.zeros((k, N))

    for i in range(k):
        logScaledComp[i, :] = loggausspdf_t(xp, mu[i], Sigma[:, :, i]) + torch.log(alpha[i])

    max_term, _ = torch.max(logScaledComp, dim=0)

    logpdf = max_term + torch.log(torch.sum(torch.exp(logScaledComp - max_term.unsqueeze(0)), dim=0))

    return logpdf.squeeze()


class NormalizingFlowModel(nn.Module):

    def __init__(self, args, current_prop):
        super().__init__()
        self.device = args.device
        self.flows = nn.ModuleList([RealNVP(dim=args.dim) for _ in range(args.n_layer)]).to(self.device)

        for f in self.flows:
            f.zero_initialization()

        self.args = args
        self.current_prop = current_prop
        self.mu_ = nn.Parameter(torch.from_numpy(current_prop.mean), requires_grad=True).to(self.device)

    def forward_(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        return x, log_det

    def forward(self, samples):
        samples = samples + self.mu_.unsqueeze(1)
        samples_ = samples.reshape([-1, self.args.dim])

        samples_nf, log_det = self.forward_(samples_)
        log_det = log_det.reshape([self.args.N, self.args.K])

        log_target = self.args.log_target_t(samples_nf, self.args).reshape([self.args.N, self.args.K])

        if self.args.weighting == 'Standard':
            log_proposal = torch.zeros([self.args.N, self.args.K])
            for n in range(self.args.N):
                log_proposal[n, :] = loggausspdf_t(samples[n, :, :], self.mu_[n, :], torch.from_numpy(self.current_prop.cov[:, :, n]) )

        elif self.args.weighting == 'DM':
            log_proposal = logGMMpdf_t(samples_, self.mu_,
                                       torch.from_numpy(self.current_prop.cov)).reshape([self.args.N, self.args.K])

        log_proposal = log_proposal - log_det
        return samples_nf.reshape(samples.size()), log_target - log_proposal

