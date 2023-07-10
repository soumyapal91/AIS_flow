import numpy as np
from scipy.linalg import solve, cholesky

from utils import *
from utils_torch import *


def log_target_t(xp, args):
    if xp.ndim == 1:
        xp = xp.unsqueeze(0)

    if args.example == 'Gaussian':
        return loggausspdf_target_t(xp, torch.from_numpy(args.target_mu), torch.from_numpy(args.target_cov), torch.from_numpy(args.target_cov_inv), torch.tensor(args.target_cov_logdet))

    elif args.example == 'Banana':
        yp = torch.clone(xp)
        yp[:, [1]] = yp[:, [1]] + args.target_b * (yp[:, [0]] ** 2 - args.target_c ** 2)
        return loggausspdf_target_t(yp, torch.zeros_like(xp), torch.from_numpy(args.target_cov_trans), torch.from_numpy(args.target_cov_trans_inv), torch.tensor(args.target_cov_trans_logdet))

    elif args.example == 'GMM':
        return logGMMpdf_target_t(xp, torch.from_numpy(args.target_mu), torch.from_numpy(args.target_cov), torch.from_numpy(args.target_alpha), torch.from_numpy(args.target_cov_inv), torch.from_numpy(args.target_cov_logdet))

    elif args.example == 'Logistic':
        log_prior = loggausspdf_target_t(xp, torch.zeros_like(xp), torch.from_numpy(args.prior_cov), torch.from_numpy(args.prior_cov_inv), torch.tensor(args.prior_cov_logdet))

        prob = torch.sigmoid(torch.mm(torch.from_numpy(args.X), torch.transpose(xp, 0, 1)))
        prob = torch.clamp(prob, 1e-14, 1 - 1e-14)

        labels = torch.tile(torch.from_numpy(args.Y[None, :].T), [1, xp.shape[0]])
        ll = torch.sum(labels * torch.log(prob) + (1 - labels) * torch.log(1 - prob), dim=0)
        return log_prior + ll


def log_target(xp, args):
    if xp.ndim == 1:
        xp = xp[None, :]

    if args.example == 'Gaussian':
        return loggausspdf_target(xp, args.target_mu, args.target_cov, args.target_cov_inv, args.target_cov_logdet)

    elif args.example == 'Banana':
        yp = xp.copy()
        yp[:, [1]] = yp[:, [1]] + args.target_b * (yp[:, [0]] ** 2 - args.target_c ** 2)
        return loggausspdf_target(yp, np.zeros_like(xp), args.target_cov_trans, args.target_cov_trans_inv, args.target_cov_trans_logdet)

    elif args.example == 'GMM':
        return logGMMpdf_target(xp, args.target_mu, args.target_cov, args.target_alpha, args.target_cov_inv, args.target_cov_logdet)

    elif args.example == 'Logistic':
        log_prior = loggausspdf_target(xp, np.zeros_like(xp), args.prior_cov, args.prior_cov_inv, args.prior_cov_logdet)

        prob = 1.0 / (1 + np.exp(-np.dot(args.X, xp.T)))
        prob = np.clip(prob, 1e-14, 1-1e-14)
        labels = np.tile(args.Y[None, :].T, [1, xp.shape[0]])
        ll = np.sum(labels * np.log(prob) + (1 - labels) * np.log(1 - prob), axis=0)
        return log_prior + ll


def grad_neg_hess_inv_log_target(xp, args):
    # xp must be one sample
    if args.example == 'Gaussian':
        grad_ = loggauss_grad(xp, args.target_mu, args.target_cov_inv)
        neg_hess_inv = args.target_cov

    elif args.example == 'Banana':
        yp = xp.copy()
        yp[1] = yp[1] + args.target_b * (yp[0] ** 2 - args.target_c ** 2)

        grad_ = -xp / np.diag(args.target_cov_trans)
        grad_[0] = grad_[0] - 2 * args.target_b * yp[0] * yp[1] / args.target_cov_trans[1, 1]
        grad_[1] = -yp[1] / args.target_cov_trans[1, 1]

        neg_hess_ = np.diag(1.0 / np.diag(args.target_cov_trans))
        neg_hess_[0, 0] = neg_hess_[0, 0] + (2 * args.target_b * yp[1] + (2 * args.target_b * yp[0]) ** 2) / \
                          args.target_cov_trans[1, 1]
        neg_hess_[0, 1] = 2 * args.target_b * yp[0] / args.target_cov_trans[1, 1]
        neg_hess_[1, 0] = neg_hess_[0, 1]

        neg_hess_inv = np.zeros_like(neg_hess_)

        if is_pos_def(neg_hess_[0:2, 0:2]):
            try:
                neg_hess_inv[0:2, 0:2] = np.linalg.inv(neg_hess_[0:2, 0:2])
            except:
                neg_hess_inv[0:2, 0:2] = np.inf * np.eye(2)
        else:
            neg_hess_inv[0:2, 0:2] = np.inf * np.eye(2)

        neg_hess_inv[2:, 2:] = args.target_cov_trans[2:, 2:]

    elif args.example == 'GMM':
        alpha_posterior = posterior_latent_GMM(xp, args.target_mu, args.target_cov, args.target_alpha, args.target_cov_inv, args.target_cov_logdet)

        grad_s = np.array([loggauss_grad(xp, args.target_mu[i, :], args.target_cov_inv[:, :, i]) for i in range(args.target_nc)])

        grad_ = np.dot(grad_s.T, alpha_posterior)

        term1 = np.sum(alpha_posterior[i] * args.target_cov_inv[:, :, i] for i in range(args.target_nc))
        term2 = np.dot(grad_[None, :].T, grad_[None, :])
        term3 = -np.sum(alpha_posterior[i] * np.dot(grad_s[i, :][None, :].T, grad_s[i, :][None, :]) for i in range(args.target_nc))

        neg_hess_ = term1 + term2 + term3

        if is_pos_def(neg_hess_):
            try:
                neg_hess_inv = np.linalg.inv(neg_hess_)
            except:
                neg_hess_inv = np.inf * np.eye(args.dim)
        else:
            neg_hess_inv = np.inf * np.eye(args.dim)

    elif args.example == 'Logistic':
        grad_log_prior = loggauss_grad(xp, np.zeros_like(xp), args.prior_cov_inv)

        prob = 1.0 / (1 + np.exp(-np.dot(args.X, xp)))
        prob = np.clip(prob, 1e-14, 1-1e-14)

        grad_ll = np.sum((args.Y - prob)[:, None] * args.X, axis=0)
        grad_ = grad_log_prior + grad_ll

        term1 = args.prior_cov_inv
        term2 = np.dot(args.X.T, (prob * (1-prob))[:, None] * args.X)
        neg_hess_ = term1 + term2

        if is_pos_def(neg_hess_):
            try:
                neg_hess_inv = np.linalg.inv(neg_hess_)
            except:
                neg_hess_inv = np.inf * np.eye(args.dim)
        else:
            neg_hess_inv = np.inf * np.eye(args.dim)

    return grad_, neg_hess_inv


def grad_log_target(xp, args):
    # xp must be one sample
    if args.example == 'Gaussian':
        grad_ = loggauss_grad(xp, args.target_mu, args.target_cov_inv)

    elif args.example == 'Banana':
        yp = xp.copy()
        yp[1] = yp[1] + args.target_b * (yp[0] ** 2 - args.target_c ** 2)

        grad_ = -xp / np.diag(args.target_cov_trans)
        grad_[0] = grad_[0] - 2 * args.target_b * yp[0] * yp[1] / args.target_cov_trans[1, 1]
        grad_[1] = -yp[1] / args.target_cov_trans[1, 1]

    elif args.example == 'GMM':
        alpha_posterior = posterior_latent_GMM(xp, args.target_mu, args.target_cov, args.target_alpha, args.target_cov_inv, args.target_cov_logdet)

        grad_ = np.dot(np.array([loggauss_grad(xp, args.target_mu[i, :], args.target_cov_inv[:, :, i]) for i in range(args.target_nc)]).T, alpha_posterior)

    elif args.example == 'Logistic':
        grad_log_prior = loggauss_grad(xp, np.zeros_like(xp), args.prior_cov_inv)

        prob = 1.0 / (1 + np.exp(-np.dot(args.X, xp)))
        prob = np.clip(prob, 1e-14, 1-1e-14)

        grad_ll = np.sum((args.Y - prob)[:, None] * args.X, axis=0)
        grad_ = grad_log_prior + grad_ll

    return grad_
