import pandas as pd
from tqdm import tqdm
from time import sleep
import numpy as np
import itertools
import json, os, re, sys
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import ParameterGrid, RepeatedKFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy import sparse

from src.python.utils import *

class RunElastic:
    def __init__(self, df, seed=0):
        self.df = df
        self.seed = seed
        self.targets = ["Care", "Fairness", "Loyalty", "Authority", "Purity"]

    def build(self, inputs_name):
        cols = list()
        for c in self.df.columns:
            for name in inputs_name:
                if c.startswith(name + "."):
                    cols.append(c)
        """ Normalize feature columns that represent word counts """
        for c in cols:
            if "count" in c:
                self.df[c] = self.df[c] / self.df["doc_len"]
        X = self.df.loc[:, cols]
        Y = self.df.loc[:, self.targets]
        return X, Y

    @ignore_warnings(category=ConvergenceWarning)
    def run_repeated(self, feature_prefix, n_trials=10, kfold_num=5):

        if type(feature_prefix) == str:
            feature_prefix = [feature_prefix]
        X, Y = self.build(feature_prefix)
        X = X.values

        folder = RepeatedKFold(n_splits=kfold_num,
                               n_repeats=n_trials,
                               random_state=self.seed)
        results = list()
        desc = "+".join(feature_prefix)
        cv_iterator = tqdm(folder.split(X), total=n_trials * kfold_num, ncols=50, desc=desc)
        for train_index, test_index in cv_iterator:
            for f in Y.columns:
                y = Y[f].values
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = ElasticNetCV(random_state=self.seed,
                                     n_alphas=50, cv=10, n_jobs=4,
                                     l1_ratio=[.01, .1, 0.3, .5, 0.7, 0.9, 0.99],
                                     selection='random',
                                     tol=5e-3, verbose=0)

                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)

                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                cv_iterator.set_description("{} {}: {:.2f}".format(desc, f, test_r2))
                cv_iterator.refresh()
                sleep(0.01)

                r_row = {"foundation": f, "test_r2": test_r2, "train_r2": train_r2, "alpha": model.alpha_,
                        "l1_ratio": model.l1_ratio_}
                results.append(r_row)
        df = pd.DataFrame(results)
        return df


def load_data(load_path, min_docs):

    df = pd.read_csv(load_path, '\t')
    df = df[df.num_posts >= min_docs]
    # normalize lda docs (after aggregation in previous step)
    lda_cols = [c for c in df.columns if c.startswith("lda.")]
    if len(lda_cols) > 0:
        df.loc[:, lda_cols] = StandardScaler().fit_transform(df.loc[:, lda_cols].values)
    return df

def run(feature_set, level, overwrite=True, min_docs=10):

    corpus = load_data("./data/features/public_{}_dataset.tsv".format(level), min_docs=min_docs)
    print(corpus)
    worker = RunElastic(corpus)

    output_dir = "./output/regression_results/{}_{}".format(level, min_docs)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, "{}.csv".format(feature_set))
    if os.path.exists(path) and not overwrite:
        raise ValueError("No iterations run; file exists")
    stats = worker.run_repeated([feature_set])
    stats.to_csv(path)

if __name__ == '__main__':

    subject_feature_sets = "BOM lda mfd.ddr mfd2.ddr liwc.ddr mfd.count mfd2.count liwc.count".split()
    doc_feature_sets = ["BERT"]

    # Main analysis:
    for feature_set in subject_feature_sets:
        run(feature_set=feature_set, level='subject')
    for feature_set in doc_feature_sets:
        run(feature_set=feature_set, level='doc')

    # Full analysis (supplemental materials)
    """
    for feature_set in subject_feature_sets+doc_feature_sets:
        for level in ['doc', 'subject']:
            for d in [1,25]:
                run(feature_set=feature_set, level=level, overwrite=True, min_docs=d)
    """

