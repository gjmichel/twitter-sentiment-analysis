import pandas as pd
import numpy as np
import re

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from nltk.corpus import stopwords
stp = stopwords.words('english')




stopwords = list(STOP_WORDS)
parser = English()


def get_time_window(df, start_date, end_date) : 
  mask = (df['date_days'] >= start_date) & (df['date_days'] <= end_date)
  df2 = df.loc[mask]
  return df2



# according to https://arxiv.org/pdf/1904.07531.pdf, removing stopwords has no real impact in performance even when using contextualized embeddings
# we tried both approaches and found no difference



def spacy_tokenizer(sentence):
    # we don't remove stopwords here, tokenizer for BERT
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = " ".join([i for i in mytokens])
    mytokens = re.sub(r"[@]+", "", mytokens).strip() # remove @
    mytokens = re.sub('#', '', mytokens) #remove #
    mytokens = re.sub('RT[\s]+', '', mytokens) #remove RT
    mytokens = re.sub('https?://\S+|www\.\S+', '', mytokens) # remove http links
    return mytokens


# newStopWords contains a list of words for which embeddings (in case of Word2Vec algorithm) are not helping clustering 
newStopWords = ['said','tell', 'says', 'amp', 'want', 'maybe', 'told', 'hope', 'yes', 'know', 'think', 'great', 'week', 'want', 'real', 'biggest', 'big', 'today', 'way', 'real', 'look', 'lol', 'watch', 'like', 'ago', 'sure', 'thing', 'good', 'better', 'likely', 'getting', 'lot', 'high', '&', 'got', 'start', 'started', 'going', 'new', 'according', 'needs', 'taking', 'guy', 'day', 'use', 'may', 'say', 'bad', 'time','goldmansachs']
stopwords_ext = stp + newStopWords

def tokenizer_extended_stopwords(sentence):
    mytokens = parser(sentence)
   
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
     
    mytokens = [ word for word in mytokens if word not in stopwords_ext]
    mytokens = " ".join([i for i in mytokens])
    mytokens = re.sub(r"[@]+", "", mytokens).strip() # remove @
    mytokens = re.sub('#', '', mytokens) #remove #
    mytokens = re.sub('RT[\s]+', '', mytokens) #remove RT
    mytokens = re.sub('https?://\S+|www\.\S+', '', mytokens) # remove http links
    return mytokens



def remove_company_from_tweets(sentence,company) : 
  mytokens = parser(sentence)
  mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
  mytokens = [ word for word in mytokens if word not in stopwords_ext]
  mytokens = " ".join([i for i in mytokens])

  words_to_remove = get_words_to_remove(company)

  for word in words_to_remove: 
      mytokens = re.sub(word, '', mytokens)

  return mytokens


    
def get_words_to_remove(company): 

  if company == 'MorganStanley': 
    word_list = ['morganstanley', 'morgan', 'stanley', 'ms']
    return word_list

  elif company == 'BankOfAmerica':
    word_list = ['bankofamerica', 'bankofamericamerilllynch', 'baml']
    return word_list
  
  elif company == 'JPMorgan':
    word_list = ['jpmorgan', 'jpmorganchase', 'jp', 'jpmc']
    return word_list

  elif company == 'GoldmanSachs':   
    word_list = ['goldmansachs', 'goldman', 'sachs', 'gs']
    return word_list

  elif company == 'Wealthfront':   
    word_list = ['wealthfront', 'wf']
    return word_list


#start_dates = ['2019-01-01','2019-03-01','2019-05-01','2019-07-01','2019-09-01','2019-11-01','2020-01-01','2020-03-01','2020-05-01']
#end_dates = ['2019-03-01','2019-05-01','2019-07-01','2019-09-01','2019-11-01','2019-01-01','2020-03-01','2020-05-01','2020-07-01']

# uncomment above when good data available
start_dates = ['2019-05-01','2019-07-01','2019-09-01','2019-11-01','2020-01-01','2020-03-01','2020-05-01']
end_dates = ['2019-07-01','2019-09-01','2019-11-01','2019-01-01','2020-03-01','2020-05-01','2020-07-01']



def preprocessing(df, embeddings_model, company):


  df['date_days'] = pd.to_datetime(df['date']).dt.date.astype(str)
  df['text'] = df["text"].astype(str)
  df = df.reset_index().drop(columns = 'index').reset_index()


  if embeddings_model == 'Word2Vec':
    # remove stopwords, and words related to a company
    df['processed_text'] = df["text"].apply(tokenizer_extended_stopwords)
    df['processed_text'] = df["processed_text"].apply(remove_company_from_tweets, args = (company,))


  elif embeddings_model == 'BERT':
    # we keep all the stopwords
    df['processed_text'] = df["text"].apply(spacy_tokenizer)

  return df
















