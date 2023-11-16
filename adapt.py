import numpy as np
import scipy, torch, random
import time

from utils import *
from utils_torch import *
from target_dist import *
from initialization import *
from scipy.special import gamma


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


def gramis_adapt(particles, logW, current_prop, args, epoch):
    adapted_prop = current_prop

    for n in range(args.N):
        ill_cond = False
        starting_point = current_prop.mean[n, :]
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

        distances = current_prop.mean[n, :] - current_prop.mean
        distances = distances / (np.sqrt(np.sum(distances ** 2, axis=1)) + 1e-6).reshape(-1, 1)

        adapted_prop.mean[n, :] = new_location + (0.99 ** epoch) * step_size * distances.mean(axis=0)
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


def hmc_adapt(current_prop, args, burnin=False):
    adapted_prop = current_prop

    mu_ = np.zeros([args.N, args.dim])

    for n in range(args.N):
        mu_[n, :] = leapfrog(current_prop.mean[n, :], args, M=np.eye(args.dim), L=args.L_hmc, step_size=args.eps_hmc, burnin=burnin)

    log_w_ = logGMMpdf(mu_, current_prop.mean, current_prop.cov)

    indx = resample(args.N, log_w_)
    adapted_prop.mean = mu_[indx, :]

    return adapted_prop


def leapfrog(xp, args, M, L, step_size, burnin):
    if isdiag(M):
        M_inv = np.diag(1.0 / np.diag(M))
    else:
        M_inv = np.linalg.inv(M)

    if burnin:
        xp_current = xp.copy()
        for _ in range(args.burn_in):
            q = np.dot(scipy.linalg.cholesky(M, lower=True), np.random.normal(size=args.dim))
            q_current = q.copy()

            lop_H_current = 0.5 * np.sum(q_current * np.dot(M_inv, q_current)) - args.log_target(xp_current, args)

            q = q + 0.5 * step_size * args.grad_log_target(xp, args)

            for i in range(L):
                xp = xp + step_size * np.dot(M_inv, q)
                if i < L-1:
                    q = q + step_size * args.grad_log_target(xp, args)

            q = q + 0.5 * step_size * args.grad_log_target(xp, args)

            lop_H_prop = 0.5 * np.sum(q * np.dot(M_inv, q)) - args.log_target(xp, args)

            if np.random.uniform() < np.min([1, np.exp(lop_H_current - lop_H_prop)]):
                xp_current = xp.copy()

        return xp_current

    else:
        xp_current = xp.copy()

        q = np.dot(scipy.linalg.cholesky(M, lower=True), np.random.normal(size=args.dim))
        q_current = q.copy()

        lop_H_current = 0.5 * np.sum(q_current * np.dot(M_inv, q_current)) - args.log_target(xp_current, args)

        q = q + 0.5 * step_size * args.grad_log_target(xp, args)

        for i in range(L):
            xp = xp + step_size * np.dot(M_inv, q)
            if i < L - 1:
                q = q + step_size * args.grad_log_target(xp, args)

        q = q + 0.5 * step_size * args.grad_log_target(xp, args)

        lop_H_prop = 0.5 * np.sum(q * np.dot(M_inv, q)) - args.log_target(xp, args)

        if np.random.uniform() < np.min([1, np.exp(lop_H_current - lop_H_prop)]):
            xp_current = xp.copy()

        return xp_current

