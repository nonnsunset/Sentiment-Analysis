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

gen_all_df = pd.read_csv("KiiteSentiment/kiite_dataset.csv")
gen_train_df, gen_valid_df = train_test_split(gen_all_df, test_size=0.15, random_state=1412)
Edu_all_df = pd.read_csv("KiiteSentiment/Education_Kiite.csv")
Edu_train_df, Edu_valid_df = train_test_split(Edu_all_df, test_size=0.15, random_state=1412)
Love_all_df = pd.read_csv("KiiteSentiment/love_Kiite.csv")
Love_train_df, Love_valid_df = train_test_split(Love_all_df, test_size=0.15, random_state=1412)

#lm data
gen_data_lm = load_data(model_path, "kiite_lm.pkl")
gen_data_lm.sanity_check()
Edu_data_lm = load_data(model_path, "Edukiite_lm.pkl")
Edu_data_lm.sanity_check()
Love_data_lm = load_data(model_path, "Lovekiite_lm.pkl")
Love_data_lm.sanity_check()

#classification data
tt = Tokenizer(tok_func=ThaiTokenizer, lang="th", pre_rules=pre_rules_th, post_rules=post_rules_th)
processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=gen_data_lm.vocab, max_vocab=60000, min_freq=20)]
Edu_processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=Edu_data_lm.vocab, max_vocab=60000, min_freq=20)]
Love_processor = [TokenizeProcessor(tokenizer=tt, chunksize=10000, mark_fields=False),
            NumericalizeProcessor(vocab=Love_data_lm.vocab, max_vocab=60000, min_freq=20)]

gen_data_cls = (ItemLists(model_path,train=TextList.from_df(gen_train_df, model_path, cols=["texts"], processor=processor),
                     valid=TextList.from_df(gen_valid_df, model_path, cols=["texts"], processor=processor))
    .label_from_df("tags")
    .databunch(bs=50)
    )
Edu_data_cls = (ItemLists(model_path,train=TextList.from_df(Edu_train_df, model_path, cols=["texts"], processor=Edu_processor),
                     valid=TextList.from_df(Edu_valid_df, model_path, cols=["texts"], processor=Edu_processor))
    .label_from_df("tags")
    .databunch(bs=50)
    )
Love_data_cls = (ItemLists(model_path,train=TextList.from_df(Love_train_df, model_path, cols=["texts"], processor=Love_processor),
                     valid=TextList.from_df(Love_valid_df, model_path, cols=["texts"], processor=Love_processor))
    .label_from_df("tags")
    .databunch(bs=50)
    )
gen_data_cls.sanity_check()
Edu_data_cls.sanity_check()
Love_data_cls.sanity_check()

#model
config = dict(emb_sz=400, n_hid=1550, n_layers=4, pad_token=1, qrnn=False,
             output_p=0.4, hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)
trn_args = dict(bptt=70, drop_mult=0.7, alpha=2, beta=1, max_len=500)

gen_learn_data = text_classifier_learner(gen_data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
Edu_learn_data = text_classifier_learner(Edu_data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)
Love_learn_data = text_classifier_learner(Love_data_cls, AWD_LSTM, config=config, pretrained=False, **trn_args)

#load pretrained finetuned model
gen_learn_data.load_encoder('kiite_enc')
Edu_learn_data.load_encoder('Edukiite_enc')
Love_learn_data.load_encoder('Lovekiite_enc')

gen_learn_data.load('Kiite_Model')
Edu_learn_data.load('Education_Model')
Love_learn_data.load('Love_Model')

def convertTub(a):
    if str(a) == "pos":
        return "pos"
    elif str(a) == "neg":
        return "neg"
    else:
        return "ERROR: Unknown Type"

from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/')
def Home():  
   resp = {
       "status":200,
       "message":"OK",
   }
   return jsonify(resp)

@app.route('/General')
def predict():
    text = request.args.get('text')
    predicted = gen_learn_data.predict(text)
    a = convertTub(predicted[0])
    resp = {
        "status" : 200,
        "message": "OK",
        "data":{
            "text": text,
            "mood": a,
        }
    }
    return jsonify(resp),200

@app.route('/Education')
def Ed_predict():
    text = request.args.get('text')
    predicted = Edu_learn_data.predict(text)
    a = convertTub(predicted[0])
    resp = {
        "status" : 200,
        "message": "OK",
        "data":{
            "text": text,
            "mood": a,
        }
    }
    return jsonify(resp),200

@app.route('/Love')
def Lo_predict():
    text = request.args.get('text')
    predicted = Love_learn_data.predict(text)
    a = convertTub(predicted[0])
    resp = {
        "status" : 200,
        "message": "OK",
        "data":{
            "text": text,
            "mood": a,
        }
    }
    return jsonify(resp),200
   

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5000, debug=True)