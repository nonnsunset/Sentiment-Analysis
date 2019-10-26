from google.colab import drive
drive.mount('/content/gdrive')
root_dir = "/content/gdrive/My Drive/"
##your current directory will be /content/"
!ls

# #uncomment if you are running from google colab
# !pip install sklearn_crfsuite
# !pip install --pre pythainlp
# !pip install fastai
# !pip install emoji
# !mkdir "/content/gdrive/My Drive/KiiteSentiment"; ls

import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm_notebook
from collections import Counter
import re

from sklearn.model_selection import train_test_split

#viz
import matplotlib.pyplot as plt
import seaborn as sns

from fastai.text import *
from fastai.callbacks import CSVLogger, SaveModelCallback

from pythainlp.ulmfit import *

model_path = '/content/gdrive/My Drive/KiiteSentiment' #your path#

#use pandas to read your data(CSV file)
all_df = pd.read_csv("/content/gdrive/My Drive/Dataset_kiite/kiite_dataset.csv")
train_df, valid_df = train_test_split(all_df, test_size=0.15, random_state=1412)

#Show five column
all_df.head()

tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=None, max_vocab=60000, min_freq=2)]

data_lm = (TextList.from_df(all_df, model_path, cols="texts", processor=processor)
    .split_by_rand_pct(valid_pct = 0.01, seed = 1412)
    .label_for_lm()
    .databunch(bs=48))
data_lm.sanity_check()
data_lm.save('/content/gdrive/My Drive/kiite_lm.pkl') ##แปลงข้อมูล Tokenize, Numericalize ให้เรียบร้อย แล้วเซฟไว้ก่อน (databunch_languagemodel )

data_lm.sanity_check()
len(data_lm.train_ds), len(data_lm.valid_ds)
print(len(data_lm.train_ds), len(data_lm.valid_ds))

config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False, tie_weights=True, out_bias=True,
             output_p=0.25, hidden_p=0.1, input_p=0.2, embed_p=0.02, weight_p=0.15)
trn_args = dict(drop_mult=1., clip=0.12, alpha=2, beta=1)

learn = language_model_learner(data_lm, AWD_LSTM, config=config, pretrained=False, **trn_args)

#load pretrained models
learn.load_pretrained(**_THWIKI_LSTM)

learn.predict('สังคมไทยนั้น', 50, temperature=0.5)

#train frozen
print("training frozen")
learn.freeze_to(-1)
learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

##Checkpoint วันหลังจะได้แค่มาโหลด ไม่ต้อง train
learn.save('/content/gdrive/My Drive/learner_language_model-1e-3')
learn.load('/content/gdrive/My Drive/learner_language_model-1e-3');

#train unfrozen
print('training unfrozen')
learn.unfreeze()
# learn.fit_one_cycle(10, 1e-4, moms=(0.8, 0.7)) 
learn.fit_one_cycle(1, 1e-4, moms=(0.8, 0.7)) ##สำหรับทดสอบ

learn.save('/content/gdrive/My Drive/kiite_lm')
learn.save_encoder('/content/gdrive/My Drive/kiite_enc')

learn.predict('ผมอยาก', 50, temperature=0.5)

#lm data
data_lm = load_data(model_path, "/content/gdrive/My Drive/kiite_lm.pkl")
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
print(len(data_cls.vocab.itos))

#model
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
             output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

learn = text_classifier_learner(data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
#load pretrained finetuned model
learn.load_encoder("/content/gdrive/My Drive/kiite_enc")

#train unfrozen
learn.freeze_to(-1)
learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7))
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))
learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7),
                    callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name='/content/gdrive/My Drive/Kiite_Models/Kiite_Model')])
#Predict#######