import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

algs = ['PMC', 'GR-PMC', 'LR-PMC', 'SL-PMC', 'O-PMC', 'GRAMIS', 'HAIS', 'VAPIS', 'NF-PMC']

for f in csv_files:
    print('----------------------------------------------------------------------------')
    df = pd.read_csv(f, sep=',', header=0)
    print('File Name:', f.split("\\")[-1])

    results = np.array(df.values)

    for i in range(len(algs)-1):
        if np.mean(results[:, i]) > np.mean(results[:, -1]):
            res = wilcoxon(results[:, i], results[:, -1], alternative='greater')
            print(algs[i] + '>' + algs[-1] + ', p value:' + str(res.pvalue*100) + '%')
        else:
            res = wilcoxon(results[:, i], results[:, -1], alternative='less')
            print(algs[i] + '<' + algs[-1] + ', p value:' + str(res.pvalue*100) + '%')


