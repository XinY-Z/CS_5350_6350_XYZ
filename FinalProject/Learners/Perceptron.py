from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from FinalProject.LanguageModel.TfIdf import dataset
import numpy as np


## Instantiate perceptron algorithm
clf = SGDClassifier(
    loss='perceptron',
    random_state=1,
    average=True
)

## Use stratified k-fold cross validation to split dataset (k=10)
dataset_x = dataset.iloc[:, :-1]
dataset_y = dataset.iloc[:, -1]
skf = StratifiedKFold(n_splits=10)
'''np.random.seed(123)
train_index = np.random.choice(dataset.shape[0], dataset.shape[0] * 9 // 10, replace=False)
test_index = [index for index in range(dataset.shape[0]) if index not in train_index]'''

## train averaged perceptron and return error rates
train_errors, test_errors = [], []
for train_index, test_index in skf.split(dataset_x, dataset_y):
    train_x, test_x = dataset_x[train_index], dataset_x[test_index]
    train_y, test_y = dataset_y[train_index], dataset_y[test_index]

    clf.fit(train_x, train_y)
    train_error = 1 - clf.score(train_x, train_y)
    test_error = 1 - clf.score(test_x, test_y)
    train_errors.append(train_error)
    test_errors.append(test_error)

print(f'Training error: {np.mean(train_errors)}')
print(f'Test error: {np.mean(test_errors)}')
