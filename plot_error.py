import numpy as np
import pickle as pk
import random, scipy
# from scipy.stats import bootstrap
import time
import matplotlib.pyplot as plt
# import seaborn as sns

example = 'GMM'
dim = 200

num_trials = 100

J = 500
N = 100
K = 10

learn_var = False

algs = ['DM-PMC(local)', 'VAPIS', 'NF-PMC']
sigma_props = [1.0] #[1.0, 2.0, 3.0]

legend_all = []

fig, ax = plt.subplots()
pp = []

for alg in algs:
    for sigma_prop in sigma_props:

        filename = "results/{}/{}_dim={}_sigma_prop={}_ntrial={}_J={}_N={}_K={}_learn_var={}.pk".format(example,
              alg, dim, sigma_prop, num_trials, J, N, K, learn_var)

        errors = pk.load(open(filename, "rb"))

        metric = []
        runtime = []
        for err in errors:
            runtime.append(err.runtime)
            metric.append(err.mse_mu)
            # print(alg)
            # print(err.mse_mu)

        runtime = np.array(runtime).mean(axis=0)
        metric = np.array(metric)

        # p, = ax.semilogy(np.arange(0, 500, 1) + 1, metric.mean(axis=0)[np.arange(0, 500, 1)])
        p, = ax.plot(np.arange(0, 500, 1) + 1, metric.mean(axis=0)[np.arange(0, 500, 1)])
        pp.append(p)
        # rng = np.random.default_rng()
        # res_all = [bootstrap([x, ], np.mean, confidence_level=0.95, random_state=rng) for x in acc_all]
        # ax.fill_between(alpha_all, [res.confidence_interval.low for res in res_all], [res.confidence_interval.high for res in res_all], alpha=.25)

        # rng = np.random.default_rng()
        # res_all = [stats.bootstrap([metric[:, i], ], np.mean, confidence_level=0.95, random_state=rng) for i in np.arange(0, 500, 1)]
        # ax.fill_between([res.confidence_interval.low for res in res_all], [res.confidence_interval.high for res in res_all], alpha=.25)

        legend_all.append(alg + ' ($\sigma$=' + str(sigma_prop) + ')')

        print(alg + ' ($\sigma$=' + str(sigma_prop) + ')')
        print(metric.mean(axis=0)[-1])
        # print(runtime[-1]/500)

ax.legend(pp, legend_all, loc='upper right', shadow=True)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.rcParams["font.size"] = "20"
# plt.xlim((1, 500))
plt.show()
# plt.title('Dataset:' + dataset + ', Base model: ' + model)

# fig.tight_layout()
# filename__ = dir_fig + '/' + dataset + '_' + model + '.png'
# fig.savefig(filename__, bbox_inches=Bbox([[-0.5, -0.5], fig.get_size_inches()]), dpi=250)
# plt.close()