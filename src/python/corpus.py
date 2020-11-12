import os
import ntpath
import json
import re
from itertools import product
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

from src.python.utils import *
from src.python.bert_helpers import *

class Features:

    def __init__(self, path_to_processed_tsv):
        self.df = pd.read_csv(path_to_processed_tsv, '\t')
        return

    def __load_dictionary(self, dic_file_path):

        d_name = ntpath.basename(dic_file_path).split('.')[0]
        loaded = read_dic_file(dic_file_path)
        words, stems = dict(), dict()
        for cat in loaded:
            words[cat] = list()
            stems[cat] = list()
            for word in loaded[cat]:
                if word.endswith('*'):
                    stems[cat].append(word.replace('*', ''))
                else:
                    words[cat].append(word)
        rgxs = dict()
        for cat in loaded:
            name = "{}.{}".format(d_name,cat)
            if len(stems[cat]) == 0:
                regex_str = r'\b(?:{})\b'.format("|".join(words[cat]))
            else:
                unformatted = r'(?:\b(?:{})\b|\b(?:{})[a-zA-Z]*\b)'
                regex_str = unformatted.format("|".join(words[cat]),
                        "|".join(stems[cat]))
            rgxs[name] = re.compile(regex_str)
        return rgxs, words, stems

    def count(self, dic_file_path):

        vectors = list()
        names = list()
        rgxs, _, _ = self.__load_dictionary(dic_file_path)
        for cat in rgxs:
            bow = CountVectorizer(token_pattern=rgxs[cat])\
                .fit(self.df.text.values)
            #vocab = bow.get_feature_names()
            X = bow.transform(self.df.text.values).sum(axis=1)
            self.df["{}.count.{}".format(cat[:cat.find('.')], cat[cat.find('.')+1:])] = np.squeeze(np.asarray(X))
        return

    def _avg_vecs(self, words, E, embed_size=300, max_size=None, min_size=1):
        vecs = list()
        for w in words:
            if w in E:
                vecs.append(E[w])
            if max_size is not None:
                if len(vecs) >= max_size:
                    break
        if len(vecs) < min_size:
            empty_array = np.empty((embed_size,))
            empty_array[:] = np.NaN
            return empty_array
        return np.array(vecs).mean(axis=0)

    def _dictionary_centers(self, d_path, d_name, vec_name):
        _, d_words, _ = self.__load_dictionary(d_path)

        vocab = list(set([w for cat in d_words for w in d_words[cat]]))
        path = PRETRAINED[vec_name]
        E, vec_size = load_glove(path, vocab)

        names = list()
        vecs = list()
        for category in d_words:
            vecs.append(self._avg_vecs(d_words[category],
                    E, embed_size=vec_size, max_size=25))
            names.append("{}.ddr.{}".format(d_name, category))
        return np.array(vecs, dtype=np.float32), names

    def embeddings(self, glove_path, dictionary_paths=None):

        docs = self.df["text"].values.tolist()
        bow = CountVectorizer(docs, min_df=2, tokenizer=tokenize).fit(docs)
        vocab = bow.get_feature_names()
        E, vec_size = load_glove(glove_path, vocab)

        corpus_means = np.array([self._avg_vecs(tokenize(x.lower()), E) for x in self.df.text.values])
        corpus_means_df = pd.DataFrame(corpus_means, index=self.df.index,
                columns=["BOM.{}".format(i) for i in range(corpus_means.shape[1])])
        self.df = pd.concat((self.df, corpus_means_df), axis=1, sort=False)

        if dictionary_paths is not None:
            for d_path in dictionary_paths:
                d_name = ntpath.basename(d_path).split('.')[0]
                centers, names = self._dictionary_centers(d_path, d_name, "glove")
                sims = cosine_matrix(corpus_means, centers)
                sims = pd.DataFrame(sims, columns=names, index=self.df.index)
                self.df = self.df.join(sims)
        return

    def bert(self, batch_size=256, max_seq_len=100):

        def preprocess(docs, max_size):

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            docs = [" ".join(x.split()[:max_size]) for x in docs]
            max_size = max(max_size, max([len(x.split()) for x in docs]))

            return tokenizer, max_size

        texts = self.df["text"].values.tolist()
        tokenizer, max_len = preprocess(texts, max_seq_len)

        model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        X = prepareBertInput(tokenizer, texts, max_len)

        bert_embed = np.zeros((len(texts), 768))
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))
            batch = texts[start_idx:end_idx]
            input_ids = [X[0][start_idx:end_idx], X[1][start_idx:end_idx], X[2][start_idx:end_idx]]
            outputs = model(input_ids)
            layer4 = outputs[-1][-4][:, 1:, :]
            layer3 = outputs[-1][-3][:, 1:, :]
            layer2 = outputs[-1][-2][:, 1:, :]
            layer1 = outputs[-1][-1][:, 1:, :]
            avg = tf.keras.layers.Average()
            avg_embedding = avg([layer4, layer3, layer2, layer1])
            embedding = tf.math.reduce_mean(avg_embedding, axis=1)
            bert_embed[start_idx: end_idx, :] = embedding

        feature_length = bert_embed.shape[-1]
        bert_vecs = pd.DataFrame(bert_embed, index=self.df.index,
                columns=["BERT.{}".format(i) for i in range(feature_length)])
        self.df = pd.concat((self.df, bert_vecs), axis=1, sort=False)
        return

    def lda_features(self, agg_type, lda_prefix, tokenized_docs=None):

        lda_model = load(os.path.join(lda_prefix, "saved_model.pkl"))
        id2word = Dictionary.load_from_text(os.path.join(lda_prefix, "id2word"))

        if agg_type == 'doc':

            # retrieve and save document topic probabilities
            DT_gen = lda_model.read_doctopics(lda_model.fdoctopics())
            DT = np.array([[v for idx, v in doc] for doc in DT_gen])
            dt_df = pd.DataFrame(DT, columns=["lda.{}".format(c) for c in range(DT.shape[1])])
            self.df = pd.concat((self.df, dt_df), axis=1, sort=False)

        elif agg_type == 'subject' and subject_tokens is not None:

            # collect each subject's word frequencies, normalize, and
            #       then calculate p(topic|subject) for each topic
            #subject_tokens = tokenized_docs.apply(lambda x: [a for b in x for a in b])
            subject_bow = subject_tokens.apply(lambda doc: id2word.doc2bow(doc))
            subject_bow = corpus2dense(subject_bow.values.tolist(),
                    num_terms=len(id2word),
                    num_docs=len(subject_bow)).T
            s_bow = pd.DataFrame(subject_bow, index=subject_tokens.index) \
                    .rename(columns=id2word)

            l2_norm =np.linalg.norm(s_bow.values, ord=2, axis=1, keepdims=True)
            s_bow.loc[:, :] = s_bow.values/l2_norm
            W = lda_model.get_topics()
            prod = np.matmul(s_bow.values, W.T)
            prod_norm = prod/np.linalg.norm(prod, ord=2, axis=1, keepdims=True)
            prod_norm = np.nan_to_num(prod_norm)

            features = pd.DataFrame(prod_norm, index=s_bow.index)
            features.columns = ["lda.{}".format(c) for c in features.columns]

        self.df = pd.concat((self.df, features), axis=1, sort=False)
        return
