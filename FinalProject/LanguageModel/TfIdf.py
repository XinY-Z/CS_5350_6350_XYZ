import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer(TfidfVectorizer):

    def __init__(self, ngram, nmessage, preprocessor):
        super().__init__()
        self.__nmessage = nmessage
        self.__ngram = ngram
        self.preprocessor = preprocessor
        self.input = None
        self.doclist = []
        self.data = pd.DataFrame()

    ## load data
    def load(self, dataloader):
        self.input = dataloader.data

    ## select first x messages in each encounter as predictor (x=3)
    ## transform data structure to fit sklearn tf-idf vectorizer
    def preprocess(self):

        ## remove auto-generated messages
        self.input = self.input[self.input['originator'] != 'Auto']
        data_sel = self.input.groupby('encounterId').head(self.__nmessage)
        data_sel = data_sel.loc[:, ['messageId', 'encounterId', 'originator', 'message', 'engagement']]
        data_sel['message'] = data_sel['message'].apply(lambda x: str(x))
        doclist = data_sel.groupby('encounterId')['message'].apply(list).to_list()
        doclist = [' '.join(doc) for doc in doclist]
        ## remove redacted parts
        doclist = [re.sub('\[(.*?)]', '', doc) for doc in doclist]
        self.doclist = doclist

    ## Convert texts to vectors using tf-idf
    def text2vec(self):
        self.preprocess()

        ## instantiate tf-idf vectorizer
        if isinstance(self.__ngram, int):
            tfidf = TfidfVectorizer(ngram_range=(self.__ngram, self.__ngram),
                                    preprocessor=self.preprocessor.preprocess)
        elif isinstance(self.__ngram, tuple):
            tfidf = TfidfVectorizer(ngram_range=(self.__ngram[0], self.__ngram[1]),
                                    preprocessor=self.preprocessor.preprocess)
        else:
            raise TypeError('Argument "ngram" must be int or tuple.')

        ## Vectorize the texts
        tf_idf = tfidf.fit_transform(self.doclist)
        '''word_scores = pd.DataFrame(tf_idf[0].T.todense(), index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
        word_scores = word_scores.sort_values('TF-IDF', ascending=False)
        dictionary = word_scores.to_dict()['TF-IDF']'''

        ## Create new dataset in which texts are converted to vectors
        engagement_list = self.input.groupby('encounterId').head(1)['engagement'].to_list()
        self.data = pd.DataFrame(data=tf_idf.toarray(),
                                 columns=['x' + str(i + 1) for i in range(tf_idf.toarray().shape[1])])
        self.data['outcome'] = engagement_list
