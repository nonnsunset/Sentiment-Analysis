# from fastai.text import *
# from fastai.callbacks import CSVLogger, SaveModelCallback
# from sklearn.model_selection import train_test_split
# from pythainlp.ulmfit import *
import re
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm_notebook
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback

from pythainlp.ulmfit import *

model_path = 'KiiteSentiment'

all_df = pd.read_csv("KiiteSentiment/kiite_dataset.csv")
train_df, valid_df = train_test_split(all_df, test_size=0.15, random_state=1412)

#lm data
data_lm = load_data(model_path, "kiite_lm.pkl")
data_lm.sanity_check()

#classification data
tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=data_lm.vocab, max_vocab=60000, min_freq=20)]

data_cls = (ItemLists(model_path,train=TextList.from_df(train_df, model_path, cols=["texts"], processor=processor),
                     valid=TextList.from_df(valid_df, model_path, cols=["texts"], processor=processor))
    .label_from_df("tags")
    .databunch(bs=50)
    )
data_cls.sanity_check()
len(data_cls.vocab.itos)

#model
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
             output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
#load pretrained finetuned model
learn.load_encoder('kiite_enc')

learn.load('Kiite_Model')

a = learn.predict("ควย")
print(a[0])
