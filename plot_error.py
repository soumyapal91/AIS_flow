import numpy as np
import pickle as pk
import random
from scipy.stats import wilcoxon
import time
import matplotlib.pyplot as plt
# import seaborn as sns

example = 'Banana'
dim = 50

num_trials = 10

J = 500
N = 50
K = 20

algs = ['DM_PMC_local', 'DM_PMC_local_NF', 'SL_PMC', 'VAPIS'] #['DM_PMC_local', 'DM_PMC_local_NF', 'VAPIS', 'VAPIS_NF'] #['DM_PMC_local', 'SL_PMC', 'VAPIS']
sigma_props = [1.0] #[1.0, 2.0, 3.0]

legend_all = []

fig, ax = plt.subplots()
pp = []

for alg in algs:
    for sigma_prop in sigma_props:
        filename = "results/{}/{}_dim={}_ntrial={}_sigma_prop={}_J={}_N={}_K={}.pk".format(example,
              alg, dim, num_trials, sigma_prop, J, N, K)

        errors = pk.load(open(filename, "rb"))

        metric = []
        runtime = []
        for err in errors:
            runtime.append(err.runtime)
            metric.append(err.mse_mu)

        runtime = np.array(runtime).mean(axis=0)
        metric = np.array(metric)

        print(runtime[-1])

        p, = ax.semilogy(runtime[runtime<100], metric.mean(axis=0)[runtime<100])
        pp.append(p)
        # rng = np.random.default_rng()
        # res_all = [bootstrap([x, ], np.mean, confidence_level=0.95, random_state=rng) for x in acc_all]
        # ax.fill_between(alpha_all, [res.confidence_interval.low for res in res_all], [res.confidence_interval.high for res in res_all], alpha=.25)

        legend_all.append(alg+str(sigma_prop))

ax.legend(pp, legend_all, loc='upper left', shadow=True)
plt.show()
    # # plt.title('Dataset:' + dataset + ', Base model: ' + model)
    # plt.xlabel('Fraction of queries reviewed')
    # plt.ylabel('Collaborative accuracy')
    # plt.rcParams["font.size"] = "20"
    # plt.xlim((-0.0025, alpha_all[-1] + .0025))
    # # fig.tight_layout()
    # filename__ = dir_fig + '/' + dataset + '_' + model + '.png'
    # fig.savefig(filename__, bbox_inches=Bbox([[-0.5, -0.5], fig.get_size_inches()]), dpi=250)
    # plt.close()