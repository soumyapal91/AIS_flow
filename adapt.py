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


def langevin_adapt(particles, logW, current_prop, args):
    adapted_prop = current_prop

    for n in range(args.N):
        indx = resample(1, logW[n, :])
        starting_point = particles[n, indx, :]
        log_pdf_start_ = args.log_target(starting_point, args)
        grad_, inv_neg_hess_ = args.grad_neg_hess_inv_log_target(starting_point, args)
        shift = np.dot(inv_neg_hess_, grad_)

        if np.isnan(shift).any() or np.isinf(shift).any():
            adapted_prop.mean[n, :] = starting_point
            adapted_prop.cov[:, :, n] = (args.sigma_prop ** 2) * np.eye(args.dim)

        else:
            step_size = 1.0
            while True:
                new_location = starting_point + 0.5 * step_size * shift
                if args.log_target(new_location, args) < log_pdf_start_:
                    step_size = step_size / 2
                else:
                    break

            adapted_prop.mean[n, :] = new_location
            adapted_prop.cov[:, :, n] = step_size * inv_neg_hess_

    return adapted_prop


def vi_adapt(particles, logW, current_prop, args, G_mu):
    adapted_prop = current_prop

    for n in range(args.N):
        term1 = np.dot(np.diag(np.exp(logW[n, :] - np.max(logW[n, :])) ** 2), (particles[n, :, :] - current_prop.mean[n, :]))

        if isdiag(current_prop.cov[:, :, n]):
            g_mu_n = -1.0 * np.mean(term1, axis=0) / np.diag(current_prop.cov[:, :, n])
        else:
            g_mu_n = -1.0 * solve(current_prop.cov[:, :, n], np.mean(term1, axis=0))

        G_mu[n, :] = 0.9 * G_mu[n, :] + 0.1 * (g_mu_n ** 2)
        adapted_prop.mean[n, :] = adapted_prop.mean[n, :] - args.lr_vi * g_mu_n / np.sqrt(G_mu[n, :] + 1e-8)

    return adapted_prop, G_mu

