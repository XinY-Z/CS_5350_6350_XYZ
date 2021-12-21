from FinalProject.DataLoader import DataLoader
from FinalProject.LanguageModel.Preprocessor import Preprocessor
from FinalProject.LanguageModel.TfIdf import Vectorizer
from FinalProject.Learners.SVM import SVM

## set hyperparameters to tune the model
alpha_list = [0.00005, 0.000005, 0.00001, 0.000001]

## load the dataset
dataloader = DataLoader(r'C:\Users\XinZ\Box\SafeUT_Data\Final_Anonymization\FINAL_ANONYMIZED_SAFEUT.xlsx')
# dataloader.random_select(500)
dataloader.to_engagement(6)
print('passed 1')

## preprocess data and convert to numeric vectors
preprocessor = Preprocessor(lowercase=True, lemma=True, remove_punc=True, remove_stopwords=False)
vectorizer = Vectorizer(ngram=2, nmessage=3, preprocessor=preprocessor)
vectorizer.load(dataloader)
vectorizer.text2vec()
print('passed 2')

## train the model with vectorized data
for alpha in alpha_list:
    print(f'Now using alpha={alpha}')
    svm = SVM(alpha=alpha)
    svm.kfold(n_splits=10)
    svm.evaluate(vectorizer.data)
print('all passed')
