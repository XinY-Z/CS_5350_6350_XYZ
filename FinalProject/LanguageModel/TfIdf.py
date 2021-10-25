import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from FinalProject.LanguageModel.Preprocessor import preprocessor
from FinalProject.Descriptive import data


## instantiate tf-idf vectorizer (2-gram)
tfidf = TfidfVectorizer(ngram_range=(2,2), preprocessor=preprocessor)

## remove auto-generated messages
data = data[data['originator'] != 'Auto']

## select first x messages in each encounter as predictor (x=3)
## transform data structure to feed sklearn tf-idf
data_sel = data.groupby('encounterId').head(3)[['messageId', 'encounterId', 'originator', 'message', 'dropout']]
data_sel['message'] = data_sel['message'].apply(lambda x: str(x))
doclist = data_sel.groupby('encounterId')['message'].apply(list).to_list()
doclist = [' '.join(doc) for doc in doclist]
## remove redacted parts
doclist = [re.sub('\[(.*?)]', '', doc) for doc in doclist]

## create dictionary for words with tf-idf scores
tf_idf = tfidf.fit_transform(doclist)
'''word_scores = pd.DataFrame(tf_idf[0].T.todense(), index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
word_scores = word_scores.sort_values('TF-IDF', ascending=False)
dictionary = word_scores.to_dict()['TF-IDF']'''

## calculate tf-idf vectors for each encounter
dropout_list = data_sel.groupby('encounterId').head(1)['dropout'].to_list()
dataset = pd.DataFrame({'encounterId': data_sel['encounterId'].unique(),
                        'vec': [i for i in tf_idf.toarray()],
                        'outcome': dropout_list})
