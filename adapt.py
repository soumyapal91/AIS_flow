import numpy as np
import scipy, torch, random
import time

from utils import *
from utils_torch import *
from target_dist import *
from initialization import *


def resample_adapt(particles, logW, current_prop, args):
    adapted_prop = current_prop
    if args.resampling == 'global':
        particles_ = np.array(particles).reshape([-1, args.dim])
        logW_ = np.array(logW).reshape(-1)
        indx = resample(args.N, logW_)
        adapted_prop.mean = particles_[indx, :]
    elif args.resampling == 'local':
        for n in range(args.N):
            indx = resample(1, logW[n, :])
            adapted_prop.mean[n, :] = particles[n, indx, :]

    return adapted_prop


# def langevin_adapt(particles, logW, current_prop, args):
#     adapted_prop = current_prop
#
#     for n in range(args.N):
#         indx = resample(1, logW[n, :])
#         starting_point = particles[n, indx, :]
#         log_pdf_start_ = args.log_target(starting_point, args)
#         grad_, inv_neg_hess_ = args.grad_neg_hess_inv_log_target(starting_point, args)
#         shift = np.dot(inv_neg_hess_, grad_)
#
#         if np.isnan(shift).any() or np.isinf(shift).any():
#             adapted_prop.mean[n, :] = starting_point
#             adapted_prop.cov[:, :, n] = (args.sigma_prop ** 2) * np.eye(args.dim)
#
#         else:
#             step_size = 1.0
#             while True:
#                 new_location = starting_point + 0.5 * step_size * shift
#                 if args.log_target(new_location, args) < log_pdf_start_:
#                     step_size = step_size / 2
#                 else:
#                     break
#
#             adapted_prop.mean[n, :] = new_location
#             adapted_prop.cov[:, :, n] = step_size * inv_neg_hess_
#
#     return adapted_prop


def langevin_adapt(particles, logW, current_prop, args):
    adapted_prop = current_prop

    for n in range(args.N):
        ill_cond = False
        indx = resample(1, logW[n, :])
        starting_point = particles[n, indx, :]
        log_pdf_start_ = args.log_target(starting_point, args)
        grad_, inv_neg_hess_ = args.grad_neg_hess_inv_log_target(starting_point, args)

        shift = np.dot(inv_neg_hess_, grad_)

        if np.isnan(shift).any() or np.isinf(shift).any():
            shift = np.dot(current_prop.cov[:, :, n], grad_)
            ill_cond = True

        step_size = 1.0
        while True:
            new_location = starting_point + 0.5 * step_size * shift
            if args.log_target(new_location, args) < log_pdf_start_:
                step_size = step_size / 2
            else:
                break

        adapted_prop.mean[n, :] = new_location
        if ill_cond:
            adapted_prop.cov[:, :, n] = current_prop.cov[:, :, n]
        else:
            adapted_prop.cov[:, :, n] = step_size * inv_neg_hess_

    return adapted_prop


def newton_adapt(particles, logW, current_prop, args):
    adapted_prop = current_prop

    for n in range(args.N):
        ill_cond = False
        indx = resample(1, logW[n, :])
        starting_point = particles[n, indx, :]
        log_pdf_start_ = args.log_target(starting_point, args)
        grad_, inv_neg_hess_ = args.grad_neg_hess_inv_log_target(starting_point, args)

        shift = np.dot(inv_neg_hess_, grad_)

        if np.isnan(shift).any() or np.isinf(shift).any():
            shift = np.dot(current_prop.cov[:, :, n], grad_)
            ill_cond = True

        step_size = 1.0
        while True:
            new_location = starting_point + step_size * shift
            if args.log_target(new_location, args) < log_pdf_start_:
                step_size = step_size / 2
            else:
                break

        adapted_prop.mean[n, :] = new_location
        if ill_cond:
            adapted_prop.cov[:, :, n] = current_prop.cov[:, :, n]
        else:
            adapted_prop.cov[:, :, n] = step_size * inv_neg_hess_

    return adapted_prop


def vi_adapt(particles, logW, current_prop, args, G_mu):
    adapted_prop = current_prop

    for n in range(args.N):
        term1 = np.dot(np.diag(np.exp(logW[n, :] - np.max(logW[n, :])) ** 2), (particles[n, :, :] - current_prop.mean[n, :]))

        if isdiag(current_prop.cov[:, :, n]):
            g_mu_n = -np.mean(term1, axis=0) / np.diag(current_prop.cov[:, :, n])
        else:
            g_mu_n = -solve(current_prop.cov[:, :, n], np.mean(term1, axis=0))

        G_mu[n, :] = 0.9 * G_mu[n, :] + 0.1 * (g_mu_n ** 2)
        adapted_prop.mean[n, :] = adapted_prop.mean[n, :] - args.lr_vi * g_mu_n / np.sqrt(G_mu[n, :] + 1e-8)

    return adapted_prop, G_mu


def hmc_adapt(current_prop, args):
    adapted_prop = current_prop

    mu_ = np.zeros([args.N, args.dim])

    for n in range(args.N):
        mu_[n, :] = leapfrog(current_prop.mean[n, :], args, M=np.eye(args.dim), L=args.L_hmc, step_size=args.eps_hmc)

    log_w_ = logGMMpdf(mu_, current_prop.mean, current_prop.cov)

    indx = resample(args.N, log_w_)
    adapted_prop.mean = mu_[indx, :]

    return adapted_prop


def leapfrog(xp, args, M, L, step_size):

    q = np.dot(scipy.linalg.cholesky(M, lower=True), np.random.normal(size=args.dim))

    xp_current = xp.copy()
    q_current = q.copy()

    if isdiag(M):
        M_inv = np.diag(1.0 / np.diag(M))
    else:
        M_inv = np.linalg.inv(M)

    lop_H_current = 0.5 * np.sum(q_current * np.dot(M_inv, q_current)) - args.log_target(xp_current, args)

    q = q + 0.5 * step_size * args.grad_log_target(xp, args)

    for i in range(L):
        xp = xp + step_size * np.dot(M_inv, q)
        if i < L-1:
            q = q + step_size * args.grad_log_target(xp, args)

    q = q + 0.5 * step_size * args.grad_log_target(xp, args)

    lop_H_prop = 0.5 * np.sum(q * np.dot(M_inv, q)) - args.log_target(xp, args)

    if np.random.uniform() < np.min([1, np.exp(lop_H_current - lop_H_prop)]):
        return xp
    else:
        return xp_current



