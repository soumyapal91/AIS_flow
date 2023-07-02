import numpy as np
import random
from scipy.stats import invwishart
from target_dist import *
from utils import *


class CurrentProp:

    def __init__(self, args):

        if args.example == 'Gaussian':
            self.mean = 8 * np.random.rand(args.N, args.dim) - 4
            self.cov = np.tile((args.sigma_prop ** 2) * np.eye(args.dim)[:, :, None], [1, 1, args.N])

        elif args.example == 'Banana':
            self.mean = 8 * np.random.rand(args.N, args.dim) - 4
            self.cov = np.tile((args.sigma_prop ** 2) * np.eye(args.dim)[:, :, None], [1, 1, args.N])

        elif args.example == 'GMM':
            self.mean = 20 * np.random.rand(args.N, args.dim) - 10
            self.cov = np.tile((args.sigma_prop ** 2) * np.eye(args.dim)[:, :, None], [1, 1, args.N])


def init_params(args):
    random.seed(args.seed_init)
    np.random.seed(args.seed_init)

    if args.example == 'Gaussian':
        args.target_mu = np.zeros(args.dim)
        args.target_cov = np.eye(args.dim)
        args.target_cov_inv = np.linalg.inv(args.target_cov)
        args.target_cov_logdet = logdet(args.target_cov)

    elif args.example == 'Banana':
        args.target_b = 3.0
        args.target_c = 1.0
        args.target_cov_trans = np.eye(args.dim)
        args.target_cov_trans[0, 0] = args.target_c ** 2
        args.target_cov_trans_inv = np.linalg.inv(args.target_cov_trans)
        args.target_cov_trans_logdet = logdet(args.target_cov_trans)

    elif args.example == 'GMM':
        args.target_nc = 5
        args.target_alpha = np.random.dirichlet(10*np.ones(args.target_nc))
        args.target_mu = 20 * np.random.rand(args.target_nc, args.dim) - 10
        iw = invwishart.rvs(df=args.dim, scale=np.eye(args.dim), size=args.target_nc)
        args.target_cov = np.transpose(iw, axes=[1, 2, 0]) + 2 * np.eye(args.dim)[:, :, None]
        args.target_cov_inv = np.transpose(np.array([np.linalg.inv(args.target_cov[:, :, i]) for i in range(args.target_nc)]), axes=[1, 2, 0])
        args.target_cov_logdet = np.array([logdet(args.target_cov[:, :, i]) for i in range(args.target_nc)])

    args.seeds_all = [int(x) for x in np.random.choice(np.arange(1e6), args.num_trials, replace=False)]
    print('Seeds for all trials: ' + str(args.seeds_all))

    return args


def init_alg(args):
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

    if args.algorithm == 'SL-PMC' or args.algorithm == 'O-PMC':
        args.grad_neg_hess_inv_log_target = grad_neg_hess_inv_log_target

    if args.algorithm == 'HAIS':
        args.grad_log_target = grad_log_target

    return args
