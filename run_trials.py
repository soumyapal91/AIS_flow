import os
import random
import numpy as np
import pickle as pk

import argparse
from results import *
from ais import *

parser = argparse.ArgumentParser(description='AIS')
parser.add_argument('--step_by_step', type=bool, default=False, help='set True, shows final error only if False')
parser.add_argument('--seed_init', type=int, default=123, help='initial seed')
parser.add_argument('--num_trials', type=int, default=1, help='no. random trials')

parser.add_argument("--example", type=str, default='Logistic', choices=['Gaussian', 'Banana', 'GMM', 'Logistic'])
parser.add_argument('--dim', type=int, default=200, help='dimension of target distribution')
parser.add_argument('--sigma_prop', type=float, default=1.0, help='std. dev of proposals, try [1.0, 2.0, 3.0]')

parser.add_argument("--weighting", type=str, default='DM', choices=['DM', 'Standard'])
parser.add_argument("--resampling", type=str, default='local', choices=['local', 'global'])
parser.add_argument("--adaptation", type=str, default='HMC', choices=['Resample', 'Langevin', 'Newton', 'HMC', 'VAPIS', 'NF'])

parser.add_argument('--J', type=int, default=50, help='no. iterations')
parser.add_argument('--K', type=int, default=10, help='no. samples/proposal')
parser.add_argument('--N', type=int, default=100, help='no. proposals')

parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--n_layer_nf", type=int, default=1, help='number of normalizing flow layers')
parser.add_argument('--lr_nf', type=float, default=0.05, help='learning rate for NF-PMC only, try [0.01, 0.05, 0.1]')
parser.add_argument('--step_nf', type=int, default=100, help='step for decay in learning rate for NF-PMC only')
parser.add_argument('--gamma', type=float, default=0.1, help='factor of learning rate decay for NF-PMC only')
parser.add_argument('--learn_var', type=bool, default=False, help='whether to use learnable cov matrix in NF')
parser.add_argument("--loss", type=str, default='KL', choices=['KL', 'KLrev', 'div2'])

parser.add_argument('--lr_vi', type=float, default=0.5, help='learning rate for VAPIS only, try [0.01, 0.05, 0.1]')

parser.add_argument('--L_hmc', type=int, default=1, help='no. leapfrog steps for HAIS only')
parser.add_argument('--eps_hmc', type=float, default=0.25, help='step-size of leapfrog for HAIS only]')

args = parser.parse_args()

if args.adaptation == 'Resample' and args.resampling == 'global' and args.weighting == 'Standard':
    args.algorithm = 'PMC'
    args.J = args.J * args.K
    args.K = 1
elif args.adaptation == 'Resample' and args.resampling == 'global' and args.weighting == 'DM':
    args.algorithm = 'GR-PMC'
elif args.adaptation == 'Resample' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'LR-PMC'
elif args.adaptation == 'Langevin' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'SL-PMC'
elif args.adaptation == 'Newton' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'O-PMC'
elif args.adaptation == 'HMC' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'HAIS'
elif args.adaptation == 'VAPIS' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'VAPIS'
elif args.adaptation == 'NF' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'NF-PMC'
else:
    print('Not valid arguments!!!')
    exit(0)

random.seed(args.seed_init)
np.random.seed(args.seed_init)
args.seeds_all = [int(x) for x in np.random.choice(np.arange(1e6), args.num_trials, replace=False)]
print('Seeds for all trials: ' + str(args.seeds_all))

if __name__ == "__main__":

    results = list()

    log_dir = 'results' + '/' + args.example

    check = os.path.isdir(log_dir)
    if not check:
        os.makedirs(log_dir)
        print("created folder : ", log_dir)
    else:
        print(log_dir, " folder already exists.")

    filename = "{}/{}_dim={}_sigma_prop={}_ntrial={}_J={}_N={}_K={}_learn_var={}.pk".format(log_dir,
               args.algorithm, args.dim, args.sigma_prop, args.num_trials, args.J, args.N, args.K, args.learn_var)

    for trial in range(args.num_trials):
        args.seed = args.seeds_all[trial]
        print('-------------------------------------------------------------------------------------------------------')
        print('Trial:' + str(trial) + ', example: ' + args.example + ', dimension: ' + str(args.dim) +
              ', sigma_prop=' + str(args.sigma_prop) + ', algorithm: ' + args.algorithm +
              ', J=' + str(args.J) + ', K=' + str(args.K) + ', N=' + str(args.N))

        output, args = AIS_main(args)

        results.append(Result(output, args))

    # save results
    pk.dump(results, open(filename, "wb"))

    with open(filename, 'rb') as f:
        err_list = pk.load(f)

    for trial, err in enumerate(err_list):
        print('-------------------------------------------------------------------------------------------------------')
        print('Trial:' + str(trial))
        print('Runtime: ' + str(err.runtime[-1]))
        if args.example == 'Gaussian':
            print('MSE of mean estimate: ' + str(err.mse_mu[-1]))
            print('MSE of 2nd momemnt estimate: ' + str(err.mse_m2[-1]))

        elif args.example == 'Banana':
            print('MSE of mean estimate: ' + str(err.mse_mu[-1]))

        elif args.example == 'GMM':
            print('MSE of mean estimate: ' + str(err.mse_mu[-1]))
            print('MSE of 2nd momemnt estimate: ' + str(err.mse_m2[-1]))

        elif args.example == 'Logistic':
            print('rMSE of weight estimate: ' + str(err.mse_mu[-1]))
            print('Log-likelihood of test data: ' + str(err.ll_test[-1]))

