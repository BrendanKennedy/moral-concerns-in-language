import os

import pandas as pd
from scipy.stats import sem
from tabulate import tabulate

if __name__ == '__main__':
    dfs = list()
    dir_ = "results/subject_10/"
    for f in os.listdir(dir_):
        d = pd.read_csv(os.path.join(dir_, f))
        d["method"] = f.replace(".csv", "")
        dfs.append(d)
    dir_ = "results/doc_10/"
    for f in os.listdir(dir_):
        d = pd.read_csv(os.path.join(dir_, f))
        d["method"] = f.replace(".csv", "")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True, sort=False, axis=0)
    df.method = df.method.apply(lambda x: x.replace("regression_", ""))

    method_map = {'mfd.count': 'MFD',
                  'mfd.ddr': 'MFD$_{DDR}$',
                  'mfd2.count': 'MFD2',
                  'mfd2.ddr': 'MFD2$_{DDR}$',
                  'liwc.count': 'LIWC',
                  'liwc.ddr': 'LIWC$_{DDR}$',
                  'lda': 'LDA',
                  'BOM': 'GloVe',
                  'BERT': 'BERT'}

    df.method= df.method.map(method_map)
    def agg_str(values):
        return "{:.2f} ({:.2f})".format(100 * values.mean(), 100 * sem(values))

    agg = df.groupby(['method', 'foundation'])['test_r2'] \
            .apply(agg_str) \
            .unstack() \
            .reindex(method_map.values()) \
            .reindex(columns=["Care", "Fairness", "Loyalty", "Authority", "Purity"])

    print(tabulate(agg, headers='keys', tablefmt="latex_raw"))
