import numpy as np
import re
import langdetect
import string

remove = re.compile(r"(?:http(s)?[^\s]+|(pic\.[^s]+)|@[\s]+)")
alpha = re.compile(r'(?:[a-zA-Z\']{2,15}|[aAiI])')
printable = set(string.printable)

def load_glove(path, vocab, embed_size=300):
    E = dict()
    vocab = set(vocab)
    found = list()
    with open(path) as fo:
        for line in fo:
            tokens = line.strip().split()
            vec = tokens[len(tokens) - embed_size:]
            token = "".join(tokens[:len(tokens) - embed_size])
            if token in vocab:
                E[token] = np.array(vec, dtype=np.float32)
                found.append(token)
    if vocab is not None:
        print("Found {}/{} tokens in {}".format(len(found),
                                len(vocab), path))
    return E, embed_size

def read_dic_file(f):
    categories = dict()
    words = list()
    with open(f, 'r') as fo:
        for line in fo:
            if line.strip() == '':
                continue
            if line.startswith("%"):
                continue
            line_split = line.split()
            if line_split[0].isnumeric() and len(line_split) == 2:
                cat_id, category = line.split()
                categories[int(cat_id)] = category
            else:
                words.append(line_split)
    dictionary = {category: list() for id_, category in categories.items()}
    for line in words:
        word = line[0]
        if line[1][0].isalpha():
            continue  # multi word expression
        for cat_id in line[1:]:
            dictionary[categories[int(cat_id)]].append(word)
    return dictionary


def is_english(text):
    try:
        lang = langdetect.detect(text)
        return lang == "en"
    except langdetect.lang_detect_exception.LangDetectException:
        return False

def tokenize(t, stem=False):
    t = remove.sub('', t)
    t = "".join([a for a in filter(lambda x: x in printable, t)])
    tokens = alpha.findall(t)
    return tokens

def cosine_matrix(X, Y):
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    return np.einsum('ij,kj->ik', X, Y) / np.einsum('i,j->ij', X_norm, Y_norm)

def load_features(load_path, min_docs, prefix):

    sep = '\t' if load_path.endswith(".tsv") else ','
    df = pd.read_csv(load_path, sep=sep, index_col='subject_id')
    df = df[df.num_posts >= min_docs]

    feature_cols = [c for c in df.columns if c.startswith(prefix)]
    demo_cols = ["age", "gender"]
    mfq_cols = ["Care", "Fairness", "Loyalty", "Authority", "Purity"]

    df = df.loc[:, demo_cols+mfq_cols+feature_cols]

    if prefix.startswith("lda."):
        df.loc[:, feature_cols] = StandardScaler().fit_transform(df.loc[:, feature_cols].values)

    return df
