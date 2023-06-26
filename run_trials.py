import os
import random
import numpy as np
import pickle as pk

import argparse
from results import *
from ais import *

parser = argparse.ArgumentParser(description='AIS')
parser.add_argument("--example", type=str, default='Banana', choices=['Gaussian', 'GMM', 'Banana', 'Logistic'])
parser.add_argument('--dim', type=int, default=50, help='dimension of target distribution')
parser.add_argument('--sigma_prop', type=float, default=1.0, help='std. dev of proposals')

parser.add_argument('--J', type=int, default=500, help='no. iteration')
parser.add_argument('--K', type=int, default=20, help='no. samples/proposal')
parser.add_argument('--N', type=int, default=50, help='no. proposals')

parser.add_argument('--num_trials', type=int, default=10, help='no. random trials')
parser.add_argument('--seed_init', type=int, default=123, help='initial seed')

parser.add_argument("--weighting", type=str, default='DM', choices=['DM', 'Standard'])
parser.add_argument("--resampling", type=str, default='local', choices=['local', 'global'])
parser.add_argument("--adaptation", type=str, default='Langevin', choices=['Resample', 'Langevin', 'VAPIS', 'NF'])

parser.add_argument('--step_by_step', type=bool, default=False)

parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--n_layer", type=int, default=1, help='number of normalizing flow layers')
parser.add_argument('--lr_nf', type=float, default=0.01, help='learning rate for NF-PMC only, try [0.01, 0.05, 0.1]')

parser.add_argument('--lr_vi', type=float, default=0.05, help='learning rate for VAPIS only, try [0.01, 0.05, 0.1]')

args = parser.parse_args()

if args.adaptation == 'Resample' and args.resampling == 'global' and args.weighting == 'Standard':
    args.algorithm = 'PMC'
    args.J = args.J * args.K
    args.K = 1
elif args.adaptation == 'Resample' and args.resampling == 'global' and args.weighting == 'DM':
    args.algorithm = 'DM-PMC(global)'
elif args.adaptation == 'Resample' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'DM-PMC(local)'
elif args.adaptation == 'Langevin' and args.resampling == 'local' and args.weighting == 'DM':
    args.algorithm = 'SL-PMC'
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

    filename = "{}/{}_dim={}_ntrial={}_sigma_prop={}_J={}_N={}_K={}.pk".format(log_dir,
               args.algorithm, args.dim, args.num_trials, args.sigma_prop, args.J, args.N, args.K)

    for trial in range(args.num_trials):
        args.seed = args.seeds_all[trial]
        print('-------------------------------------------------------------------------------------------------------')
        print('Trial:' + str(trial) + ', algorithm: ' + args.algorithm + ', J=' + str(args.J) + ', K=' + str(args.K) + ', N=' + str(args.N) + ', sigma_prop=' + str(args.sigma_prop))

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

