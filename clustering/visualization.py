import numpy as np
import pandas as pd
import time
import nltk

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import string
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from collections import Counter
from nltk.util import everygrams

from wordcloud import WordCloud



from nltk.corpus import stopwords
stp = stopwords.words('english')
newStopWords = ['said','tell', 'says', 'amp', 'want', 'maybe', 'told', 'hope', 'yes', 'know', 'think', 'great', 'week', 'want', 'real', 'biggest', 'big', 'today', 'way', 'real', 'look', 'lol', 'watch', 'like', 'ago', 'sure', 'thing', 'good', 'better', 'likely', 'getting', 'lot', 'high', '&', 'got', 'start', 'started', 'going', 'new', 'according', 'needs', 'taking', 'guy', 'day', 'use', 'may', 'say', 'bad', 'time','goldmansachs']
stopwords_2 = stp + newStopWords

import spacy
from spacy.lang.en import English

parser = English()


# functions to get clusters and plot relevant words

def get_tweet_length_distribution(df):
  df2 = df.copy()
  df2['NumWords'] = df2['tweets'].apply(lambda x: len(x.split()))
  df2[['NumWords']].hist(figsize=(12, 6), bins=10, xlabelsize=8, ylabelsize=8);
  plt.title("Distributon of number of words in the data")


def get_cluster(df, cluster_id) : 
  cluster = df[df['topic_cluster']==cluster_id]
  return cluster


def get_top_n_words(corpus, n):
  vec = CountVectorizer(stop_words='english').fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0) 
  words_freq = [(word, sum_words[0, idx]) for word, idx in   vec.vocabulary_.items()]
  words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
  return words_freq[:n]


def show_top_n_words(corpus,n): 
    words = []
    word_values = []

    for i,j in get_top_n_words(corpus,n):
      words.append(i)
      word_values.append(j)
    fig, ax = plt.subplots(figsize = (14,11))
    ax.bar(range(len(words)), word_values);
    ax.set_xticks(range(len(words)));
    ax.set_xticklabels(words, rotation='vertical');
    ax.set_title(f"Top {n} words in cluster");
    ax.set_xlabel('Word');
    ax.set_ylabel('Number of occurences');

def plot_top_words(df,n_clusters ,n_words): 
  for k in range(n_clusters) : 
    cluster = get_cluster(df, k)
    show_top_n_words(cluster['tweets'], n_words)
    plt.title(f"Top {n_words} words in cluster {k}")


def get_cluster_sizes(df, n_clusters) : 
  for i in range(n_clusters) :
    cluster = get_cluster(df,i)
    print(f"Number of tweets in cluster {i}", cluster.shape[0])



# Visualize important n-grams

def get_all_tokens(tweet_list):
    # concat entire corpus
    all_text = ' '.join((t for t in tweet_list))
    # tokenize
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    stopwords_2.extend(['morgan','stanley','realdonaldtrump','morganstanley','ca','gt'])
    mytokens = parser(all_text)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords_2 and word not in string.punctuation]
    return mytokens



def get_top_n_grams(corpus, length_grams,number_grams):
    tokens=get_all_tokens(corpus)
    top_grams=Counter(everygrams(tokens,min_len=length_grams, max_len=length_grams+1))
    return top_grams.most_common(number_grams)


def show_top_n_grams(corpus, length_grams,number_grams): 
    n_grams = []
    n_grams_values = []
    for i,j in get_top_n_grams(corpus, length_grams,number_grams):
        n_grams.append(i[0]+' '+i[1])
        n_grams_values.append(j)

    fig, ax = plt.subplots(figsize = (14,7))
    ax.bar(range(len(n_grams)), n_grams_values);
    ax.set_xticks(range(len(n_grams)));
    ax.set_xticklabels(n_grams, rotation='vertical');
    ax.set_xlabel('Word');
    ax.set_ylabel('Number of occurences');

def plot_top_n_grams(df,n_clusters,length_grams,number_grams):
    for k in range(n_clusters) : 
        cluster = get_cluster(df, k)
        show_top_n_grams(cluster['tweets'],length_grams,number_grams)
        plt.title(f"Top {number_grams} words in cluster {k}")


        

# get custom colormaps

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('YlGnBu')
new_cmap_1 = truncate_colormap(cmap, 0.65, 0.95)
new_cmap_2 = truncate_colormap(cmap, 0.55, 0.95)





# WordClouds

def get_wordcloud(df, column, custom_cmap) :

  all_words = ''.join([word for word in df[column]])
  wordcloud = WordCloud(width=800, height=500, max_words = 40, stopwords = stopwords_2, max_font_size=110, background_color = None, colormap = custom_cmap, mode= "RGBA" ).generate(str(all_words))
  plt.figure(figsize=(15, 14))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis('off')
  plt.show()

def plot_wordclouds(df,n_clusters, column, custom_cmap): 
  for k in range(n_clusters) : 
    cluster = get_cluster(df, k)
    plt.subplot(n_clusters, 1, k+1)
    plt.axis('off')
    get_wordcloud(cluster, column, custom_cmap)


# TSNE


def plot_tsne_2D(embeddings_df, result_df, n_iter, n_clusters) : 

  time_start = time.time()

  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter = n_iter)
  tsne_results = tsne.fit_transform(embeddings_df.values)

  print('t-SNE done in {} seconds'.format(time.time()-time_start))

  result_df['tsne-2d-one'] = tsne_results[:,0]
  result_df['tsne-2d-two'] = tsne_results[:,1]

  cluster_ids = result_df['topic_cluster'].values
  tsne_data = pd.DataFrame({'topic' : cluster_ids})

  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x= result_df["tsne-2d-one"], y=result_df["tsne-2d-two"],
      hue= result_df['topic_cluster'],
      palette=sns.hls_palette(n_clusters, l=.5, s=.9),
      data=tsne_data,
      legend="full", 
      alpha = 0.3)







