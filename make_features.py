import re
import os
import argparse
import json
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from src.python.corpus import Features
from src.python.lda_helpers import prep_text_lda, fit_lda

warnings.simplefilter(action='ignore', category=FutureWarning)

# fill in paths to .dic files, or set them with environment variables
LIWC_PATH = os.environ["LIWC_PATH"] if "LIWC_PATH" in os.environ else "..."
MFD_PATH = os.environ["MFD_PATH"] if "MFD_PATH" in os.environ else "..."
MFD2_PATH = os.environ["MFD2_PATH"] if "MFD2_PATH" in os.environ else "..."

# fill in path to .txt GloVe file, or set with environment variable
GLOVE_PATH = os.environ["GLOVE_PATH"] if "GLOVE_PATH" in os.environ else "..."

def generate_doc_features():

    # this file is not publicly available. Must have a 'text' column with clean text
    corpus = Features("./data/processed/private_doc_features_with_text.tsv")
    print(corpus.df)

    print("-----------\n\nWord Count\n\n-----------")
    corpus.count(LIWC_PATH)
    corpus.count(MFD_PATH)
    corpus.count(MFD2_PATH)

    print("-----------\n\nLDA\n\n-----------")
    corpus.lda_features(agg_type="doc", lda_prefix=here("./output/lda_model/"))

    # embeddings: (averaged GloVe vectors) + (DDR)

    print("-----------\n\nGloVe and DDR\n\n-----------")
    corpus.embeddings(glove_path="/home/brendan/Data/glove.6B.300d.txt",
                      dictionary_paths=[LIWC_PATH, MFD_PATH, MFD2_PATH])

    print("-----------\n\nBERT\n\n-----------")
    corpus.bert(batch_size=16, max_seq_len=100)

    subject_cols = ['subject_id', 'age', 'gender', 'Care', 'Fairness',
            'Loyalty', 'Authority', 'Purity', 'num_posts']
    gb = corpus.df.groupby(subject_cols).mean().reset_index()
    gb.to_csv("./data/features/doc_dataset.csv", index=False)

def generate_subject_features():

    # this file is not publicly available. Must have a 'text' column with clean text
    corpus = Features("./data/processed/private_subject_dataset.tsv")

    corpus.count(LIWC_PATH)
    corpus.count(MFD_PATH)
    corpus.count(MFD2_PATH)

    # lda: load model (previously trained with run_lda.py)

    subject_text = corpus.df.set_index('subject_id').text
    _, subject_tokens = prep_text_lda(subject_text)

    lda_model = fit_lda(prefix=here("./output/lda_model/"))
    corpus.lda_features(agg_type="subject", lda_prefix=here("./output/lda_model/"),
            tokenized_docs=subject_tokens)

    # embeddings: (averaged GloVe vectors) + (DDR)

    corpus.embeddings(glove_path=GLOVE_PATH,
                      dictionary_paths=[LIWC_PATH, MFD_PATH, MFD2_PATH])
    corpus.bert(batch_size=16, max_seq_len=500)

    corpus.df.to_csv("./data/features/subject_dataset.csv", index=False)

if __name__ == '__main__':
    generate_doc_features()
    #generate_subject_features()
