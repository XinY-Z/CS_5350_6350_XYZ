from FinalProject.DataLoader import DataLoader
from FinalProject.LanguageModel.Preprocessor import Preprocessor
from FinalProject.LanguageModel.TfIdf import Vectorizer
from FinalProject.Learners.SVM import SVM


dataloader = DataLoader(r'D:\Users\u1318593\Downloads\FINAL_ANONYMIZED_SAFEUT.xlsx', sheet='Message 1')
# dataloader.random_select(500)
dataloader.to_dropout(6)
print('passed 1')

preprocessor = Preprocessor(lowercase=True, lemma=True, remove_punc=True, remove_stopwords=False)
vectorizer = Vectorizer(ngram=2, nmessage=3, preprocessor=preprocessor)
vectorizer.load(dataloader)
vectorizer.text2vec()
dataset = vectorizer.data
print('passed 2')

svm = SVM()
svm.kfold(n_splits=10)
svm.evaluate(dataset)
print('all passed')
