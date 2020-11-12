[preprint](https://psyarxiv.com/uqmty)
[OSF](https://osf.io/jcuqk/)

## Preliminaries

### Programming Environment

Python libraries can be installed (preferably using a virtual environment such as virtualenv or conda) using pip, with the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

Prepend `sudo` command if necessary. 

### Data

Study participants' identifying information will not be released, including their Facebook status updates and [](yourmorals.org) user IDs. Feature representations necessary to replicate Analysis 1 and Analysis 2 are provided in the above OSF link.

### File structure

`doc_text_dataset.tsv` and `subject_text_dataset.tsv` are not provided, though each file is assumed to have a `text` column containing cleaned text (type `str`).

```
+-- data
|  +-- processed
|  |  +-- doc_text_dataset.tsv  # one line per post; multiple posts per participant
|  |  +-- subject_text_dataset.tsv # one line per participant; posts concatenated 
|  +-- features
|  |  +-- public_doc_dataset.tsv # one line per participant; from "make_features.py"
|  |  +-- public_subject_dataset.tsv # one line per participant; from "make_features.py"
```

Public datasets can be downloaded from OSF.

### Producing Features

Though we do not release the original text, we provide the code used to produce the feature sets (from a given text dataset) used in both analyses. 

To produce all 9 feature sets (word counting MFD, MFD2, and LIWC; LDA; average GloVe vectors; DDR for MFD, MFD2, and LIWC; average BERT vectors), run `make_features.py` from shell, substituting paths to dictionary files (`*.dic`) and path to GloVe vectors (`*.txt`).

Before `make_features.py` can be called to produce LDA vectors, an LDA model must be fit. This can be done by calling `fit_lda.py`, again substituting the correct path to a file containing a `text` column (with cleaned text). This requires Mallet to be [downloaded](http://mallet.cs.umass.edu/download.php) and an environment variable `MALLET_PATH` set in `~/.bashrc` or `~/.profile`. 

GloVe features require [downloading](https://nlp.stanford.edu/projects/glove/) the text file(s) containing GloVe vectors. The vectors from Wikipedia+Gigaword (6B.300d) were used in this paper. 

### Analysis 1

Regressions for each feature set can be run from shell with `python study1_regressions.py`. To replicate analyses contained in the supplemental materials (varying minimum number of posts for removing participants; aggregation of posts) uncomment the relevant section in `study1_regressions.py`. 

Post hoc tests of the resulting explanatory coefficients are contained in `analysis1_posthoc.R`.

### Analysis 2

Exploratory analyses of the influence of moral concerns on produced categories of language (MFD2 ~ moral concerns, LIWC ~ moral concerns, LDA ~ moral concerns) are found in `analysis2.R`.
