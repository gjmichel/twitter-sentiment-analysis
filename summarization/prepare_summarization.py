
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import time
import string


import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline


nltk.download('stopwords')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = stopwords.words('english')



##  Helper functions 

def get_cluster_summarization(df, cluster_id) : 
  cluster = df[df['topic_cluster']==cluster_id]
  #print('cluster shape', cluster.shape)
  return cluster

def concatenate_tweets(tweets) :  # modify to add a dot only if the tweet hasn't one
  concatenated = ''
  punct = ['.', '?', '!']
  for tweet in tweets : 
    if list(tweet)[-1] not in punct :
    	tweet = tweet + '.'
    concatenated = concatenated + ' ' + tweet
  return concatenated



## Sentiment analysis using Vader or DistillBERT

sid = SentimentIntensityAnalyzer()

def Classify(score, treshold) : 
  if score > treshold : 
    return 'Positive'
  elif score < -treshold : 
    return 'Negative'
  else : 
    return 'Neutral'

def vader_sentiment_analysis(df,column, treshold):

  summary = {"positive":0,"neutral":0,"negative":0}
  scores_vader = []
  start = time.time()

  for x in df[column].values.astype(str): 
      score = sid.polarity_scores(x)
      scores_vader.append(score)
      if score["compound"] == 0.0: 
          summary["neutral"] +=1
      elif score["compound"] > 0.0:
          summary["positive"] +=1
      else:
          summary["negative"] +=1

  scores_vader_df = pd.DataFrame(scores_vader)
  scores_vader_df.rename(columns = {'neg':'Negative', 'neu':'Neutral', 'pos':'Positive', 'compound': 'Compound'}, inplace = True)
  scores_vader_df['Sentiment'] = scores_vader_df['Compound'].apply(lambda x : Classify(x,treshold))
  scores_vader_df['original_tweet'] = df[column].values
  print('Time taken to classify {} tweets : {} sec\n'.format(df.shape[0],time.time() - start))

  return scores_vader_df



def get_cluster_sentiments(df, n_clusters, column, treshold): 

  # gather each cluster in a separate dataframe
  cluster_list = [f"cluster_{k}" for k in range(n_clusters)]
  cluster_dict = {name: get_cluster_summarization(df,k) for k, name in enumerate(cluster_list)}

  # compute sentiment for each tweet and sentiment type counts for each cluster
  sentiment_dict = {}
  sentiment_counts = []
  for i in cluster_dict : 
    sentiment_dict[i] = vader_sentiment_analysis(cluster_dict[i], column, treshold)
    sentiment_counts.append(pd.value_counts(sentiment_dict[i]['Sentiment']))
  sentiment_counts_df = pd.DataFrame(sentiment_counts, columns = ['Neutral', 'Positive', 'Negative'], index = sentiment_dict.keys())
  return sentiment_dict, sentiment_counts_df



def hf_sentiment_analyzer(df, column) : 
  tweets = df[column].values.astype(str)
  labels = []
  scores = []
  for tweet in tweets : 
    sentiment = sentiment_analyzer(tweet)
    label = sentiment[0]['label']
    score = sentiment[0]['score']
    labels.append(label)
    scores.append(score)
  result_df = pd.DataFrame({'labels':labels, 'scores': scores, 'original_tweet' : tweets})
  return result_df


def df_sentiment_analyser(df, n_clusters, column) : 

  cluster_list = [f"cluster_{k}" for k in range(n_clusters)]
  cluster_dict = {name: get_cluster_summarization(df,k) for k, name in enumerate(cluster_list)}

  sentiments_dict = {}
  for i in cluster_dict : 
    sentiments_dict[i] = hf_sentiment_analyzer(cluster_dict[i], column)

  return sentiments_dict
    


## Summarization preprocessing, without reordering tweets

  

def get_positive_negative_tweets(SentimentAnalyzer, df, n_clusters, column, treshold) : 

  #lists of dataframe names containing positive or negative tweets for each cluster
  cluster_list_positive = [f"cluster_{k}_positive" for k in range(n_clusters)]
  cluster_list_negative = [f"cluster_{k}_negative" for k in range(n_clusters)]

  positive_sentiment_dict = {}
  negative_sentiment_dict = {}

  # get sentiment dictionnary from previous function
  if SentimentAnalyzer == 'Vader':

    sentiment_dict = get_cluster_sentiments(df, n_clusters, column, treshold)[0]
    for i in sentiment_dict:

      dataframe = sentiment_dict[i]

      positive_df_ = dataframe.loc[dataframe['Sentiment'] == 'Positive']
      positive_df = positive_df_.sort_values(by ='Compound', ascending = False)
      positive_sentiment_dict[i] = positive_df

      negative_df_ = dataframe.loc[dataframe['Sentiment'] == 'Negative']
      negative_df = negative_df_.sort_values(by ='Compound') # ascending = True as scores are negative here 
      negative_sentiment_dict[i] = negative_df

  elif SentimentAnalyzer == 'DistillBert': 
    sentiment_dict = df_sentiment_analyser(df, n_clusters, column)
   
    for i in sentiment_dict:
      dataframe = sentiment_dict[i]

      positive_df_ = dataframe.loc[dataframe['labels'] == 'POSITIVE']
      positive_df = positive_df_.sort_values(by ='scores', ascending = False)
      positive_sentiment_dict[i] = positive_df

      negative_df_ = dataframe.loc[dataframe['labels'] == 'NEGATIVE']
      negative_df = negative_df_.sort_values(by ='scores', ascending = False) 
      negative_sentiment_dict[i] = negative_df


  return positive_sentiment_dict, negative_sentiment_dict


def sequential_concatenation(SentimentAnalyzer, df, n_clusters, column, treshold, n_tweets) : 
  
  positive_dict, negative_dict = get_positive_negative_tweets(SentimentAnalyzer, df, n_clusters, column, treshold)
  concatenated_positive = {}
  concatenated_negative = {}

  for i in positive_dict :
    relevant_slice_p =  positive_dict[i][column].values.astype(str)[:n_tweets]
    concatenated_positive[i] = concatenate_tweets(relevant_slice_p)

  for i in negative_dict :
    relevant_slice_n =  negative_dict[i][column].values.astype(str)[:n_tweets]
    concatenated_negative[i] = concatenate_tweets(relevant_slice_n)

  return concatenated_positive, concatenated_negative




## Summarization preprocessing, reordered tweets


# keep the punctuation since it may be relevant for similarity in sentiment
def clean_string(sentence): 
  sentence = ''.join([word for word in sentence]).lower()
  s = ' '.join(([word for word in sentence.split() if word not in stop_words]))
  return s

def cosine_similarity_calculator(sentences) :
  cleaned = list(map(clean_string, sentences))
  similarity_vectorizer = TfidfVectorizer().fit_transform(cleaned)  
  similarity_vectors = similarity_vectorizer.toarray()
  cosine_sim = cosine_similarity(similarity_vectorizer.toarray())
  return cosine_sim


# represent all the similarities as [i,j,similarity(i,j)] for i<j
def from_matrix_to_array(cosine_matrix) : 

  L = []
  for i in range(cosine_matrix.shape[0]) : 
    for j in range(i) : 
      L.append([i,j, cosine_matrix[i,j]])
  L = np.array(L)
  return L


# order the tweets according to their similarity 
def get_ordered_indices(cosine_matrix) : 

    L = from_matrix_to_array(cosine_matrix)
    L_sorted = L[L[:,2].argsort()[::-1]]
    index_1 = int(L_sorted[0,0])
    index_2 = int(L_sorted[0,1])
    L_sorted = np.delete(L_sorted, (0), axis = 0)

    val_1 = np.min(np.where(np.logical_or(L_sorted[:,0]==index_1, L_sorted[:,1]==index_1)))
    val_2 = np.min(np.where(np.logical_or(L_sorted[:,0]==index_2, L_sorted[:,1]==index_2)))
    if val_1 > val_2 : 
      index_1, index_2 = index_2, index_1
    ordered_indices  = [index_1, index_2]

    to_remove = []
    for p in range (len(L_sorted)) : 
        if (L_sorted[p,0] == index_2)| (L_sorted[p,1] == index_2)| (L_sorted[p,0] == index_1)| (L_sorted[p,1] == index_1):
            to_remove.append(p)
    L_sorted = np.delete(L_sorted, (to_remove), axis = 0)


    for k in range(cosine_matrix.shape[0]-3) :
      index_1 = int(L_sorted[0,0])
      index_2 = int(L_sorted[0,1])
      L_sorted = np.delete(L_sorted, (0), axis = 0)
      if (index_1 in L_sorted[:,0])|(index_1 in L_sorted[:,0]) : 
        val_1 = np.min(np.where(np.logical_or(L_sorted[:,0]==index_1, L_sorted[:,1]==index_1)))
      else : val_1 = 1000
        
      if (index_2 in L_sorted[:,0])|(index_2 in L_sorted[:,0]) : 
        val_2 = np.min(np.where(np.logical_or(L_sorted[:,0]==index_2, L_sorted[:,1]==index_2)))
      else : val_2 = 1000
      
      if val_1 > val_2 : 
        index_1, index_2 = index_2, index_1
      ordered_indices.append(index_2)

      to_remove = []
      for p in range (len(L_sorted)) : 
        if (L_sorted[p,0] == index_2)| (L_sorted[p,1] == index_2):
          to_remove.append(p)
      L_sorted = np.delete(L_sorted, tuple(to_remove), axis = 0)  # remove all the rows where one of the tweets involved has already been added to the sorted list of tweets


    for l in range(cosine_matrix.shape[0]):
      if l not in ordered_indices : 
        ordered_indices.append(l)
    return ordered_indices


def order_your_tweets(cosine_matrix, tweets): 
  ordered_indices = get_ordered_indices(cosine_matrix)
  ordered_tweets = []
  for i in ordered_indices : 
      ordered_tweets.append(tweets[i])
  return ordered_tweets




# take the n_relevant_tweets with stronger sentiment
# then, based on similarity, we'll select just n_tweets (if a tweet is weird but has a strong sentiment, we won't take it)

def custom_tweets_order(SentimentAnalyzer, df, n_clusters, column, treshold, n_relevant_tweets) : 

  positive_sentiment_dict, negative_sentiment_dict = get_positive_negative_tweets(SentimentAnalyzer, df, n_clusters, column, treshold)

  positive_sentiment_ordered = {}
  negative_sentiment_ordered = {}

  for i in positive_sentiment_dict : 
    if positive_sentiment_dict[i]['original_tweet'].values.astype(str).shape[0] >5 : 
      cosine_matrix = 100*cosine_similarity_calculator(positive_sentiment_dict[i]['original_tweet'].values.astype(str)[:n_relevant_tweets])
      ordered_tweets_p = order_your_tweets(cosine_matrix, positive_sentiment_dict[i]['original_tweet'].values.astype(str)[:n_relevant_tweets])
      positive_sentiment_ordered[i] = ordered_tweets_p
    else : 
      positive_sentiment_ordered[i] = positive_sentiment_dict[i]['original_tweet'].values.astype(str)

  for i in negative_sentiment_dict : 
    if negative_sentiment_dict[i]['original_tweet'].values.astype(str).shape[0] >5 : 
      cosine_matrix = 100*cosine_similarity_calculator(negative_sentiment_dict[i]['original_tweet'].values.astype(str)[:n_relevant_tweets])
      ordered_tweets_n = order_your_tweets(cosine_matrix, negative_sentiment_dict[i]['original_tweet'].values.astype(str)[:n_relevant_tweets])
      negative_sentiment_ordered[i] = ordered_tweets_n
    else : 
      negative_sentiment_ordered[i] = negative_sentiment_dict[i]['original_tweet'].values.astype(str)

  return positive_sentiment_ordered, negative_sentiment_ordered



def sequential_concatenation_custom(SentimentAnalyzer, df, n_clusters, column, treshold, n_relevant_tweets, n_tweets) : 
  
  positive_dict, negative_dict = custom_tweets_order(SentimentAnalyzer, df, n_clusters, column, treshold, n_relevant_tweets)
  concatenated_positive = {}
  concatenated_negative = {}

  for i in positive_dict :
    relevant_slice_p =  positive_dict[i][:n_tweets]
    concatenated_positive[i] = concatenate_tweets(relevant_slice_p)

  for i in negative_dict :
    relevant_slice_n =  negative_dict[i][:n_tweets]
    concatenated_negative[i] = concatenate_tweets(relevant_slice_n)

  return concatenated_positive, concatenated_negative







def summarization_preprocessing_df(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_relevant_tweets, n_tweets): 

	if tweets_ordering == 'Initial' : 
			concatenated_positive, concatenated_negative = sequential_concatenation(SentimentAnalyzer, df, n_clusters, column, treshold, n_tweets) 

	elif tweets_ordering == 'Custom' : 
			concatenated_positive, concatenated_negative = sequential_concatenation_custom(SentimentAnalyzer, df, n_clusters, column, treshold, n_relevant_tweets, n_tweets)

	return concatenated_positive, concatenated_negative




def prepare_summarization_complete(df_list,SentimentAnalyzer, tweets_ordering, n_clusters, column, treshold, n_relevant_tweets, n_tweets ): 

	positive_data = []
	negative_data = []

	if SentimentAnalyzer == 'DistillBert': 
		sentiment_analyzer = pipeline('sentiment-analysis', model = 'distilbert-base-uncased-finetuned-sst-2-english', tokenizer = 'distilbert-base-uncased-finetuned-sst-2-english', framework = 'tf')

	for df in df_list: 
		concatenated_positive, concatenated_negative = summarization_preprocessing_df(SentimentAnalyzer, tweets_ordering, df, n_clusters, column, treshold, n_relevant_tweets, n_tweets)
		positive_data.append(concatenated_positive)
		negative_data.append(concatenated_negative)

	return positive_data, negative_data










