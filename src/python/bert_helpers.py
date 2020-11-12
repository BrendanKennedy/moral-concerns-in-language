import tensorflow as tf
from transformers import *
import random as rn
import numpy as np
from tqdm import tqdm
import os, json, sys
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.random.seed(1)
rn.seed(1)

def prepareBertInput(tokenizer, docsChunk, max_seq_length):
    idsChunk, masksChunk, segmentsChunk = [], [], []
    for doc in tqdm(docsChunk, desc="Converting docs to features"):
        tokens = tokenizer.tokenize(doc)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0 : (max_seq_length - 2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(ids)
        # Zero-pad up to the sequence length.
        while len(ids) < max_seq_length:
            ids.append(0)
            masks.append(0)
        segments = [0] * max_seq_length
        assert len(ids) == max_seq_length
        assert len(masks) == max_seq_length
        assert len(segments) == max_seq_length
        idsChunk.append(ids)
        masksChunk.append(masks)
        segmentsChunk.append(segments)
    encodedChunk = [idsChunk, masksChunk, segmentsChunk]
    encodedChunk = [tf.keras.backend.cast(x, dtype="int32") for x in encodedChunk]
    return encodedChunk
