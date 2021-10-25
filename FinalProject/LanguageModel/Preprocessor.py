import spacy


nlp = spacy.load('en_core_web_sm')

# remove punctuation and stopwords, lowercase the words and lemmatize
def preprocessor(string, lowercase=True, lemma=True, remove_punc=True, remove_stopwords=False):
    if lowercase:
        string = string.lower()
    doc = nlp(string)

    if remove_punc:
        doc = [token for token in doc if not token.is_punct]
    if remove_stopwords:
        doc = [token for token in doc if not token.is_stop]

    if lemma:
        doc = [token.lemma_ for token in doc]
    else:
        doc = [token.text for token in doc]

    return ' '.join(doc)