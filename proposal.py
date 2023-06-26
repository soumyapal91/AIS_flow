import numpy as np
import scipy, torch, random
import time
from tqdm import tqdm
from utils import *
from utils_torch import *
from target_dist import *
from initialization import *


def proposal_step(current_prop, args):
    samples = np.array([(np.tile(current_prop.mean[n, :], [args.K, 1]) +
                        np.dot(scipy.linalg.cholesky(current_prop.cov[:, :, n], lower=True), np.random.normal(size=[args.dim, args.K])).T) for
                        n in range(args.N)])

    return samples


def importance_weighting(samples, current_prop, args):

    samples_ = samples.reshape([-1, args.dim])

    log_target = args.log_target(samples_, args).reshape([args.N, args.K])

    if args.weighting == 'Standard':
        log_proposal = np.zeros([args.N, args.K])
        for n in range(args.N):
            log_proposal[n, :] = loggausspdf(samples[n, :, :], current_prop.mean[n, :], current_prop.cov[:, :, n])

    elif args.weighting == 'DM':
        log_proposal = logGMMpdf(samples_, current_prop.mean, current_prop.cov).reshape([args.N, args.K])

    return log_target - log_proposal