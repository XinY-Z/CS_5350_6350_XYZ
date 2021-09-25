## Import packages
import pandas as pd
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
    for attribute in dataset:
        if is_numeric_dtype(dataset[attribute]):
            dataset = num2bin(dataset, attribute)
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
def num2bin(dataset, attribute):
    median = dataset[attribute].median()
    dataset[attribute] = dataset[attribute].apply(lambda i: 'Left' if i <= median else 'Right')
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
def id3(train_dir, test_dir, metric, max_depth):
    train = load_csv(train_dir)
    test = load_csv(test_dir)
    tree = learn(train, metric, max_depth)
    predictions = list()
    for index in test.index:
        case = test.iloc[index]
        prediction = predict(tree, case)
        predictions.append(prediction)
    return predictions

## evaluate algorithm
def evaluate(train_dir, test_dir, algorithm, *args):
    predictions = algorithm(train_dir, test_dir, *args)
    predicted = [pred[0] for pred in predictions]
    test = load_csv(test_dir)
    actual = test.iloc[:, -1].to_list()
    score = 100 - accuracy(actual, predicted)
    return score
