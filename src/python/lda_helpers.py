import os
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from gensim.models.wrappers import LdaMallet
from gensim import utils
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from src.python.utils import tokenize


def prep_text_lda(docs, vocab_size=20000):
    """ docs: (pd.Series str) cleaned text """

    english_stopwords = set([s.replace("\'", "") for s in stopwords.words("english")])
    tqdm.pandas(desc="Tokenizing")
    tokenized_docs = docs.progress_apply(lambda x: [w.lower() for w in tokenize(x)])

    bigram = Phrases(tokenized_docs.values.tolist())
    phraser = Phraser(bigram)
    tqdm.pandas(desc="Bigrams")
    bigrammed_docs = tokenized_docs.progress_apply(lambda tokens_: phraser[tokens_])

    id2word = Dictionary(bigrammed_docs.values.tolist())
    id2word.filter_extremes(keep_n=vocab_size, no_above=0.5)
    id2word.filter_tokens(bad_ids=[id2word.token2id[a] for a in english_stopwords if a in id2word.token2id])
    id2word.compactify()

    tqdm.pandas(desc="Cleaning")
    tokenized = bigrammed_docs.progress_apply(lambda doc_tokens: " ".join([w for w in doc_tokens if w in id2word.token2id]))
    reconst_docs = tokenized.apply(lambda x: x.split())

    return id2word, reconst_docs

def fit_lda(prefix, tokenized_docs, id2word,
            mallet_path=os.environ["MALLET_PATH"],
            num_topics=500, iterations=500):

    if not os.path.isdir(prefix):
        os.makedirs(prefix)

    if os.path.exists(os.path.join(prefix, "saved_model.pkl")):
        return utils.SaveLoad.load(os.path.join(prefix, "saved_model.pkl"))
    elif tokenized_docs is None:
        raise ValueError("LDA model not found at {}/{}".format(prefixed, "saved_model.pkl"))

    if mallet_path is None or mallet_path == "":
        raise ValueError("No mallet path specified")

    corpus = [id2word.doc2bow(tokens) for tokens in tokenized_docs.values.tolist()]

    lda_model = LdaMallet(mallet_path=mallet_path,
                          prefix=prefix,
                          corpus=corpus,
                          id2word=id2word,
                          iterations=iterations,
                          workers=4,
                          num_topics=num_topics,
                          optimize_interval=20)
    lda_model.save(os.path.join(prefix, "saved_model.pkl"))
    id2word.save_as_text(os.path.join(prefix, "id2word"))

    # save clean lda weights for later analysis
    W = lda_model.get_topics()
    W = pd.DataFrame(W).rename(columns=id2word)
    W.index = pd.Series(["lda.{}".format(i) for i in range(len(W))], name="topic_id")
    W.to_csv(os.path.join(prefix, "lda_weights.csv"))
    return lda_model

