import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import time

from prepare_summarization import *

import json 
from transformers import T5Config, T5Tokenizer, TFT5ForConditionalGeneration

model = TFT5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')


from transformers import pipeline
summarizer_bart = pipeline("summarization", framework = 'tf')
summarizer_T5 = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")




"""## **3. Functions for automatic text sumarization with T5**"""

def T5_preprocessor(text, toprint = True) : 
  preprocessed_text = text.strip().replace("\n","")
  t5_prepared_text = "summarize: " + preprocessed_text # to make T5 aware that we want to do summarization
  if toprint == True : 
    print("Original text preprocessed: \n", preprocessed_text)
  return t5_prepared_text

def T5_summarization(text, num_beams, min_length, max_lenght): 
  t5_prepared_text = T5_preprocessor(text, toprint=False)
  tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors = 'tf')
  summary_ids = model.generate( tokenized_text,
                                num_beams = num_beams,
                                no_repeat_ngram_size=2,   # no 2-gram appear twice 
                                min_length = min_length,
                                max_length = max_lenght,
                                early_stopping=True)
  
  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return output



# global function for automatic summarization
def get_cluster_summaries_T5(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_tweets, n_relevant_tweets, num_beams, min_length, max_lenght) : 

  concatenated_positive, concatenated_negative = summarization_preprocessing_df(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_relevant_tweets, n_tweets)
  positive_summaries = {}
  negative_summaries = {}
  for i in concatenated_positive :
      text_positive = concatenated_positive[i]
      positive_summaries[i] = T5_summarization(text_positive, num_beams, min_length, max_lenght)
  for i in concatenated_negative :
      text_negative = concatenated_negative[i]
      negative_summaries[i] = T5_summarization(text_negative, num_beams, min_length, max_lenght)

  return positive_summaries, negative_summaries





def get_cluster_summaries_BART(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_tweets, n_relevant_tweets, min_length, max_length, num_beams) : 

  concatenated_positive, concatenated_negative = summarization_preprocessing_df(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_relevant_tweets, n_tweets)
  positive_summaries = {}
  negative_summaries = {}
  for i in concatenated_positive :
      text_positive = concatenated_positive[i]
      positive_summaries[i] = summarizer_bart(text_positive, min_length = min_length, max_length = max_length, num_beams = num_beams)
  for i in concatenated_negative :
      text_negative = concatenated_negative[i]
      negative_summaries[i] = summarizer_bart(text_negative, min_length = min_length, max_length = max_length, num_beams= num_beams)

  return positive_summaries, negative_summaries

#* ordering tweets according to similarity and not to sentiment score before concatenation
#* selecting the `n_relevant_tweets`with stronger sentiment, and then, based on similarity between them, select `n_tweets`. The idea is that if a tweet has a strong sentiment but is not related to the others at al, it may belong to another cluster or not be relevant at all


def summarize(Summarizer, df_list, SentimentAnalyzer, tweets_ordering,  n_clusters, column, treshold, n_tweets, n_relevant_tweets, min_length, max_length, num_beams): 

	keys = [f"Window_{i+1}" for i in range(len(df_list))]
	positive_summaries_allwindows = {}
	negative_summaries_allwindows = {}

	for i in range(len(df_list)): 
		df = df_list[i]

		if Summarizer == 'T5': 
			positive_summaries, negative_summaries = get_cluster_summaries_T5(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_tweets, n_relevant_tweets, num_beams, min_length, max_lenght) 
			positive_summaries_allwindows[keys[i]] = positive_summaries
			negative_summaries_allwindows[keys[i]] = negative_summaries

		elif Summarizer == 'BART': 
			positive_summaries, negative_summaries = get_cluster_summaries_BART(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_tweets, n_relevant_tweets, min_length, max_length, num_beams)
			positive_summaries_allwindows[keys[i]] = positive_summaries
			negative_summaries_allwindows[keys[i]] = negative_summaries

	return positive_summaries_allwindows, negative_summaries_allwindows





