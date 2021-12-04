from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


class SVM(SGDClassifier, StratifiedKFold):

    def __init__(self):
        super(SGDClassifier).__init__()
        super(StratifiedKFold).__init__()
        self.clf = SGDClassifier(loss='hinge', random_state=1)
        self.skf = None

    def kfold(self, n_splits):
        self.skf = StratifiedKFold(n_splits=n_splits)

    def evaluate(self, dataset):

        ## Use stratified k-fold cross validation to split dataset
        dataset_x = dataset.iloc[:, :-1]
        dataset_y = dataset.iloc[:, -1]

        ## train SVM and return performance metrics
        train_errors, base_rates, accuracies, precisions, recalls, f1_scores = [], [], [], [], [], []
        for train_index, test_index in self.skf.split(dataset_x, dataset_y):
            train_x, test_x = dataset_x.iloc[train_index], dataset_x.iloc[test_index]
            train_y, test_y = dataset_y.iloc[train_index], dataset_y.iloc[test_index]

            self.clf.fit(train_x, train_y)
            base_rate = test_y.mean()
            train_error = 1 - self.clf.score(train_x, train_y)
            accuracy = self.clf.score(test_x, test_y)

            predictions = self.clf.predict(test_x)
            precision = precision_score(test_y, predictions)
            recall = recall_score(test_y, predictions)
            f1 = f1_score(test_y, predictions)

            base_rates.append(base_rate)
            train_errors.append(train_error)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        print(f'Training error: {np.mean(train_errors)}')
        print(f'Base rate: {np.mean(base_rates)}')
        print(f'Accuracy: {np.mean(accuracies)}')
        print(f'Precision: {np.mean(precisions)}')
        print(f'Recall: {np.mean(recalls)}')
        print(f'F1 score: {np.mean(f1_scores)}')
