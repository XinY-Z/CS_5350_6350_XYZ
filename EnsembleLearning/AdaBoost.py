## Import packages
import numpy as np
import pandas as pd
import sys
from pandas.api.types import is_numeric_dtype
from Metrics import get_metric, get_accuracy

## load a csv file
def load_csv(filepath):
    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                    'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    dataset = pd.read_csv(filepath, names=attributes)
    return dataset

## calculate sample size
def get_size(data):
    size = data.shape[0]
    return size

## calculate vote for a classifier
def get_vote(error):
    alpha = 1/2 * np.log((1-error)/error)
    return alpha

## update weights of examples
def update_weight(weight, alpha, actual, predicted):
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            weight[i] = weight[i] * np.exp(-alpha)
        else:
            weight[i] = weight[i] * np.exp(alpha)
    weight = [w / sum(weight) for w in weight]
    return weight

## calculate weighted proportions of label values in a set/subset
def get_props(data):
    weights = data.groupby('y')['weight'].sum().to_list()
    return weights

## Split numeric values into binary
def num2bin(dataset, attribute, train_values, datatype):
    if datatype == 'train':
        median = dataset[attribute].median()
        dataset[attribute] = dataset[attribute].apply(lambda i: 'Left' if i <= median else 'Right')
        train_values.update({attribute: median})
        return dataset
    elif datatype == 'test':
        dataset[attribute] = dataset[attribute].apply(lambda i: 'Left' if i <= train_values[attribute] else 'Right')
        return dataset

## Replace missing values
def impute(dataset, attribute, train_modes, datatype):
    if datatype == 'train':
        mode = dataset[attribute].mode()
        dataset[attribute] = dataset[attribute].replace(['unknown'], mode)
        train_modes.update({attribute: mode})
    elif datatype == 'test':
        dataset[attribute] = dataset[attribute].replace(['unknown'], train_modes[attribute])
    return dataset

## select the best attribute
def get_split(dataset, metric):
    outcome_col = [col for col in dataset][-1]
    ndataset = get_size(dataset)
    total_props = get_props(dataset)
    total_metric = get_metric(total_props, metric=metric)
    best_attribute, best_gain = None, 0.0
    subsets = {}

    for attribute in [attribute for attribute in dataset if attribute != outcome_col]:
        subset = dataset.groupby(attribute)
        total_subset_metric = 0.0
        for group in subset.groups:
            nsubset = get_size(subset.get_group(group))
            subset_props = get_props(subset.get_group(group))
            subset_metric = get_metric(subset_props, metric=metric)
            weighted_subset_metric = (nsubset / ndataset) * subset_metric
            total_subset_metric += weighted_subset_metric
        gain = total_metric - total_subset_metric
        if gain < 0:
            gain = 0
        if gain >= best_gain:
            best_attribute, best_gain = attribute, gain
        '''print('attr is ' + attribute)
        print('gain is ' + str(gain))'''
    best_subset = dataset.groupby(best_attribute)
    for group, subset in best_subset:
        subsets.update({group: subset.drop(best_attribute, axis=1)})
    return {'attribute': best_attribute, 'subsets': subsets, 'gain': best_gain}

## create leaf node:
def to_leaf(dataset):
    return dataset['y'].mode().to_list()

## create child splits for a node or make leaf node
def split(node, metric, max_depth, depth):
    ## if no split occurs
    if len(node['subsets']) == 1:
        key = list(node['subsets'].keys())[0]
        node[key] = to_leaf(node['subsets'][key])
        return
    ## if depth of the tree reaches the maximum
    if depth >= max_depth:
        for subset in node['subsets']:
            node[subset] = to_leaf(node['subsets'][subset])
        return
    else:
        for subset in node['subsets']:
            ## check if all labels are the same -> no need to split
            all_label = node['subsets'][subset]['y'].to_list()
            if all(all_label[0] == v for v in all_label):
                node[subset] = to_leaf(node['subsets'][subset])
            else:
                node[subset] = get_split(node['subsets'][subset], metric)
                if node[subset]['gain'] == 0:
                    node[subset] = to_leaf(node['subsets'][subset])
                else:
                    split(node[subset], metric, max_depth, depth+1)
                    # print(depth+1)

## Learn decision tree with ID3 algorithm
def learn(data, metric='entropy', max_depth=2):
    root = get_split(data, metric)
    split(root, metric, max_depth, depth=1)
    return root

## Predict label using learned decision tree
def predict(node, case):
    test_group = case[node['attribute']]
    if test_group not in node:
        return ['?']
    else:
        if isinstance(node[test_group], dict):
            return predict(node[test_group], case)
        else:
            return node[test_group]

## return predicted values using ID3
def id3(train_dir, test_dir, metric, max_depth, impute_missing=False):
    train = load_csv(train_dir)
    test = load_csv(test_dir)
    train_values = {}
    train_modes = {}
    predictions = []

    for attribute in train:
        if is_numeric_dtype(train[attribute]):
            train = num2bin(train, attribute, train_values, 'train')
        if impute_missing and 'unknown' in train[attribute].values:
            train = impute(train, attribute, train_modes, 'train')
    for attribute in test:
        if is_numeric_dtype(test[attribute]):
            test = num2bin(test, attribute, train_values, 'test')
        if impute_missing and 'unknown' in test[attribute].values:
            test = impute(test, attribute, train_modes, 'test')
    tree = learn(train, metric, max_depth)

    for index in test.index:
        case = test.iloc[index]
        prediction = predict(tree, case)
        predictions.append(prediction)
    predicted = [pred[0] for pred in predictions]
    return predicted

## return predicted values using adaboost
def adaboost(train_dir, test_dir, max_iter):
    train = load_csv(train_dir)
    test = load_csv(test_dir)
    size = get_size(train)
    train['weight'] = [1/size] * size
    iteration = 1
    trees = []
    predictions = []

    while iteration <= max_iter:
        tree = learn(train, metric='entropy', max_depth=2)
        predicted = id3(train_dir, train_dir, metric='entropy', max_depth=2)
        actual = load_csv(train_dir)['y'].to_list()
        error = 1 - get_accuracy(actual, predicted)
        vote = get_vote(error)
        updated_weight = update_weight(train['weight'], vote, actual, predicted)
        train['weight'] = updated_weight
        trees.append((vote, tree))
        iteration += 1

    for index in test.index:
        subpredictions = []
        case = test.iloc[index]
        for vote, tree in trees:
            subprediction = predict(tree, case)
            subpredictions.append((vote, subprediction))
        prediction = pd.DataFrame(subpredictions).groupby(1).sum().idxmax().to_list()
        predictions.append(prediction[0])
    return predictions

## evaluate algorithm
def evaluate(train_dir, test_dir, algorithm, *args):
    predictions = algorithm(train_dir, test_dir, *args)
    test = load_csv(test_dir)
    actual = test['y'].to_list()
    error = 1 - get_accuracy(actual, predictions)
    return error


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    test_score = evaluate(train_file, test_file, adaboost)
    print('Prediction error (%): ' + str(test_score))
