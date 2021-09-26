## Import packages
import pandas as pd
import sys
from pandas.api.types import is_numeric_dtype
from Metrics import get_metric, accuracy

## load a csv file
def load_csv(filepath):
    if 'car' in filepath:
        attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    else:
        attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                      'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    dataset = pd.read_csv(filepath, names=attributes)
    return dataset

## calculate sample size
def get_size(data):
    size = data.shape[0]
    return size

## calculate proportions of label values in a set/subset
def get_props(data):
    keys = set(data.iloc[:, -1])
    counts = {}.fromkeys(keys, 0)
    new_counts = data.iloc[:, -1].value_counts().to_dict()
    counts.update(new_counts)
    counts_list = [v for v in counts.values()]
    counts_sum = sum(counts_list)
    props = [v / counts_sum for v in counts_list]
    return props

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
    return dataset.iloc[:, -1].mode().to_list()

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
            all_label = node['subsets'][subset].iloc[:, -1].to_list()
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
def learn(data, metric, max_depth):
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

## starter
def id3(train_dir, test_dir, metric, max_depth, impute_missing = False):
    train = load_csv(train_dir)
    test = load_csv(test_dir)
    train_values = {}
    train_modes = {}
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
    predictions = list()
    for index in test.index:
        case = test.iloc[index]
        prediction = predict(tree, case)
        predictions.append(prediction)
    predicted = [pred[0] for pred in predictions]
    print_sel = input('Do you want to print out the predicted values? (Y/N): ')
    if print_sel == 'Y':
        print('Predicted values: \n')
        print(predicted)
    return predicted

## evaluate algorithm
def evaluate(train_dir, test_dir, algorithm, *args):
    predictions = algorithm(train_dir, test_dir, *args)
    test = load_csv(test_dir)
    actual = test.iloc[:, -1].to_list()
    score = 100 - accuracy(actual, predictions)
    return score


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    metric_input = input('Which metric do you use? (entropy/majority_index/gini)')
    depth_input = int(input('How many levels do you expect for the tree?'))
    impute_input = input('Do you want to impute missing values? (Y/N)')
    if impute_input == 'Y':
        test_score = evaluate(train_file, test_file, id3, metric_input, depth_input, impute_input)
    else:
        test_score = evaluate(train_file, test_file, id3, metric_input, depth_input)
    print('Prediction error (%): ' + str(test_score))
