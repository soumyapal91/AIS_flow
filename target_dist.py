import numpy as np
from scipy.linalg import solve, cholesky

from utils import *
from utils_torch import *


def log_target_t(xp, args):

    if xp.ndim == 1:
        xp = xp.unsqueeze(0)

    if args.example == 'Gaussian':
        return loggausspdf_t(xp, torch.from_numpy(args.target_mu), torch.from_numpy(args.target_cov))

    elif args.example == 'Banana':
        yp = torch.clone(xp)
        yp[:, [1]] = yp[:, [1]] + args.target_b * (yp[:, [0]] ** 2 - args.target_c ** 2)
        return loggausspdf_t(yp, torch.zeros_like(xp), torch.from_numpy(args.target_cov_trans))


def log_target(xp, args):

    if xp.ndim == 1:
        xp = xp[None, :]

    if args.example == 'Gaussian':
        return loggausspdf(xp, args.target_mu, args.target_cov)

    elif args.example == 'Banana':
        yp = xp.copy()
        yp[:, [1]] = yp[:, [1]] + args.target_b * (yp[:, [0]] ** 2 - args.target_c ** 2)
        return loggausspdf(yp, np.zeros_like(xp), args.target_cov_trans)


def grad_neg_hess_inv_log_target(xp, args):
    # xp must be one sample
    if args.example == 'Gaussian':
        if isdiag(args.target_cov):
            grad_ = -1.0 * (xp - args.target_mu) / np.diag(args.target_cov)
        else:
            grad_ = -1.0 * solve(args.target_cov, (xp - args.target_mu))

        neg_hess_inv = args.target_cov

    elif args.example == 'Banana':
        yp = xp.copy()
        yp[1] = yp[1] + args.target_b * (yp[0] ** 2 - args.target_c ** 2)

        grad_ = -1.0 * xp / np.diag(args.target_cov_trans)
        grad_[0] = grad_[0] - 2 * args.target_b * yp[0] * yp[1] / args.target_cov_trans[1, 1]
        grad_[1] = -1.0 * yp[1] / args.target_cov_trans[1, 1]

        neg_hess_ = np.diag(1.0 / np.diag(args.target_cov_trans))
        neg_hess_[0, 0] = neg_hess_[0, 0] + (2 * args.target_b * yp[1] + (2 * args.target_b * yp[0]) ** 2) / \
                          args.target_cov_trans[1, 1]
        neg_hess_[0, 1] = 2 * args.target_b * yp[0] / args.target_cov_trans[1, 1]
        neg_hess_[1, 0] = neg_hess_[0, 1]

        neg_hess_inv = np.zeros_like(neg_hess_)

        if neg_hess_[0, 0] * neg_hess_[1, 1] - neg_hess_[0, 1] ** 2 > 1e-6:
            neg_hess_inv[0:2, 0:2] = np.linalg.inv(neg_hess_[0:2, 0:2])
        else:
            neg_hess_inv[0:2, 0:2] = np.array([[np.Inf, np.Inf], [np.Inf, np.Inf]])

        neg_hess_inv[2:, 2:] = args.target_cov_trans[2:, 2:]

    return grad_, neg_hess_inv
