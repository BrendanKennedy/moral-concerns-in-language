import pandas as pd
from src.python.lda_helpers import prep_text_lda, fit_lda

if __name__ == '__main__':
    data = pd.read_csv("./data/processed/private_doc_features_with_text.tsv", '\t')

    dic, tokenized = prep_text_lda(data.text)
    lda_model = fit_lda(prefix="./output/lda_model/",
                        tokenized_docs=tokenized,
                        id2word=dic, num_topics=300)

