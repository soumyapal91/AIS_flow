import numpy as np
import pickle as pk
import pandas as pd
import random, scipy
# from scipy.stats import bootstrap
import time
import matplotlib.pyplot as plt
# import seaborn as sns

example = 'GMM'
dim = 200

num_trials = 100

J = 50
N = 100
K = 10

learn_var = False

algs = ['GR-PMC', 'LR-PMC', 'SL-PMC', 'O-PMC', 'HAIS', 'VAPIS', 'NF-PMC']
sigma_prop = 1.0  # 1.0/2.0/3.0

metric_ = 'MSE'  #'LL'

legend_all = []

fig, ax = plt.subplots()
pp = []

df = pd.DataFrame()

for alg in algs:

    filename = "results/{}/{}_dim={}_sigma_prop={}_ntrial={}_J={}_N={}_K={}_learn_var={}.pk".format(example,
          alg, dim, sigma_prop, num_trials, J, N, K, learn_var)

    errors = pk.load(open(filename, "rb"))

    metric = []
    runtime = []
    for err in errors:
        runtime.append(err.runtime[-1] - err.runtime[-2])
        if metric_ == 'MSE':
            metric.append(err.mse_mu[-1])
        else:
            metric.append(err.ll_test[-1])

    runtime = np.array(runtime)
    metric = np.squeeze(np.array(metric))
    print('----------------------------')
    print(alg)
    print(metric_ + ': &{0:.2f}'. format(np.mean(metric)) + '$\pm$' + '{0:.2f}'. format(np.std(metric)) + ' ')
    print('Runtime' + ': &{0:.2f} sec./iter.'. format(np.mean(runtime)))

    df_ = pd.DataFrame(metric, columns=[alg])
    df = pd.concat([df, df_], axis=1)

print('----------------------------')
print(df.mean(axis=0))
df.to_csv("{}_sigma_prop={}_metric={}.csv".format(example, sigma_prop, metric_), index=False)
#
# for alg in algs:
#     data = df[alg]
#     print(data)



