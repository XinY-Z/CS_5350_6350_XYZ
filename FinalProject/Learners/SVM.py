from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from FinalProject.LanguageModel.TfIdf import dataset


## Instantiate perceptron algorithm
clf = SGDClassifier(
    loss='hinge',
    random_state=1
)

## Split dataset into training (2/3 of total) and test (1/3) sets
dataset_x = dataset.iloc[:, :-1]
dataset_y = dataset.iloc[:, -1]
skf = StratifiedKFold(n_splits=10)
skf.split(dataset_x, dataset_y)

train_errors, test_errors = [], []
for train_index, test_index in skf.split(dataset_x, dataset_y):
    train_x, test_x = dataset_x[train_index], dataset_x[test_index]
    train_y, test_y = dataset_y[train_index], dataset_y[test_index]

    clf.fit(train_x, train_y)
    train_error = 1 - clf.score(train_x, train_y)
    test_error = 1 - clf.score(test_x, test_y)
    train_errors.append(train_error)
    test_errors.append(test_error)
