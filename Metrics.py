## import packages
import numpy as np

## calculate accuracy
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

## calculate information gain
def get_metric(props, metric):
    information = select_metric(props, metric=metric)
    return information

## metric selector
def select_metric(props, metric):
    if metric == 'entropy':
        entropy = sum([-v * np.log(v) for v in props if v != 0])
        return entropy
    if metric == 'majority_error':
        me = 1-max(props)
        return me
    if metric == 'gini':
        gini = 1-sum([v ** 2 for v in props])
        return gini
