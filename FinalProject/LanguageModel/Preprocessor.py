import spacy


nlp = spacy.load('en_core_web_sm')

class Preprocessor(spacy):
    def __init__(self, lowercase, lemma, remove_punc, remove_stopwords):
        super().__init__()
        self.__lowercase = lowercase
        self.__lemma = lemma
        self.__remove_punc = remove_punc
        self.__remove_stopwords = remove_stopwords

    # remove punctuation and stopwords, lowercase the words and lemmatize
    def preprocessor(self, string):
        if self.lowercase:
            string = string.lower()
        doc = nlp(string)

        if self.remove_punc:
            doc = [token for token in doc if not token.is_punct]
        if self.remove_stopwords:
            doc = [token for token in doc if not token.is_stop]

        if self.lemma:
            doc = [token.lemma_ for token in doc]
        else:
            doc = [token.text for token in doc]

        return ' '.join(doc)
