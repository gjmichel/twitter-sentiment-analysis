import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from preprocessing_functions import *


start_dates = ['2019-01-01','2019-03-01','2019-05-01','2019-07-01','2019-09-01','2019-11-01','2020-01-01','2020-03-01','2020-05-01']
end_dates = ['2019-03-01','2019-05-01','2019-07-01','2019-09-01','2019-11-01','2019-01-01','2020-03-01','2020-05-01','2020-07-01']


def get_contextualized_sentence_embeddings_BERT(sentence, tokenizer, bert_model) :

  input_sentence = tf.constant(tokenizer.encode(sentence))[None, :]
  outputs = bert_model(input_sentence)  # The last hidden-state is the first element of the output tuple (specifically computed for classification tasks)
  last_hidden_states = outputs[0].numpy()  #get a tensor of (1, N, Bert_size) shape, where N is the number of different tokens extracted from BertTokenizer
  sentence_embedding = np.mean(last_hidden_states, axis = 1) # average the word embeddings to get an array of shape (1,Bert_size) per sentence
  return sentence_embedding



def get_dataframe_embeddings_BERT(df, tweets_column,tokenizer, bert_model, bert_size) : 

  embeddings = np.zeros((df.shape[0],bert_size))
  tweets = df[tweets_column].values.astype(str)

  for k in range(df.shape[0]) : 
    sentence_embeddings = get_contextualized_sentence_embeddings_BERT(tweets[k],tokenizer, bert_model)
    embeddings[k] = sentence_embeddings

    if k%500 == 0 : 
      print(k)

  embeddings_df = pd.DataFrame(embeddings)

  return embeddings_df




class WordVecVectorizer(object):
    def __init__(self, model,df):
        self.model = model
        self.dim = 300
        self.df = df
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([np.mean([self.model[w] for w in tweet.split() if w in self.model] or [np.zeros(self.dim)], axis=0) 
        for tweet in self.df['processed_text'].values.astype(str)])
#representing each tweet by the mean of word embeddings for the words used in the cleaned tweet


def get_Word2Vec_embeddings(model, df, tweets_column):
    wtv_vect = WordVecVectorizer(model,df)
    W2V_embeddings = wtv_vect.transform(df[tweets_column])
    return W2V_embeddings



def get_all_embeddings(embeddings_type, df, tweets_column): 
  
  if embeddings_type == 'BERT_base' :
    tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model_base = TFBertModel.from_pretrained('bert-base-uncased')
    embeddings_df = get_dataframe_embeddings_BERT(df, tweets_column,tokenizer_base, bert_model_base, 768)


  elif embeddings_type == 'BERT_large': 
    tokenizer_large = BertTokenizer.from_pretrained('bert-large-cased')
    bert_model_large = TFBertModel.from_pretrained('bert-large-cased')
    embeddings_df = get_dataframe_embeddings_BERT(df, tweets_column,tokenizer_large, bert_model_large, 1024)


  else : 
    filename = 'GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(filename, binary=True) 
    embeddings_df = pd.DataFrame(get_Word2Vec_embeddings(model, df, tweets_column))

  return embeddings_df


def get_window_tweets_and_embeddings(embeddings_type,embeddings_model, df, tweets_column, company):
  df_ready = preprocessing(df, embeddings_model, company)
  embeddings_df = get_all_embeddings(embeddings_type, df_ready, tweets_column)
  df_list = []
  embeddings_list = []
  for i in range(len(start_dates)): 
    window_df = get_time_window(df_ready, start_dates[i], end_dates[i])
    relevant_rows = window_df['index'].values.tolist()
    window_embeddings= embeddings_df.iloc[relevant_rows]

    df_list.append(window_df)
    embeddings_list.append(window_embeddings)

  return df_list,embeddings_list

