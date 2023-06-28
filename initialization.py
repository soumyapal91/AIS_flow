import numpy as np
import random
from target_dist import *


class CurrentProp:

    def __init__(self, args):

        if args.example == 'Gaussian':
            self.mean = 8 * np.random.rand(args.N, args.dim) - 4
            self.cov = np.tile((args.sigma_prop ** 2) * np.eye(args.dim)[:, :, None], [1, 1, args.N])

        elif args.example == 'Banana':
            self.mean = 8 * np.random.rand(args.N, args.dim) - 4
            self.cov = np.tile((args.sigma_prop ** 2) * np.eye(args.dim)[:, :, None], [1, 1, args.N])


def Initialization(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.algorithm == 'NF-PMC':
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        args.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if args.use_cuda:
            torch.cuda.set_device(args.gpu)
            args.device = 'cuda:{}'.format(args.gpu)
        else:
            args.device = 'cpu'

        args.log_target_t = log_target_t

    args.log_target = log_target

    if args.algorithm == 'SL-PMC':
        args.grad_neg_hess_inv_log_target = grad_neg_hess_inv_log_target

    if args.example == 'Gaussian':
        args.target_mu = np.zeros(args.dim)
        args.target_cov = np.eye(args.dim)
    elif args.example == 'Banana':
        args.target_b = 3.0
        args.target_c = 1.0
        args.target_cov_trans = np.eye(args.dim)
        args.target_cov_trans[0, 0] = args.target_c ** 2
    return args
