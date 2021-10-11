import sys
from AdaBoost import *
from Metrics import get_accuracy
from Plotter import plot


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
    t_max = int(sys.argv[3])
    train_errors = []
    test_errors = []
    iterations = list(range(1, t_max+1))
    for t in iterations:
        train_error = evaluate(train_file, train_file, adaboost, t)
        test_error = evaluate(train_file, test_file, adaboost, t)
        train_errors.append(train_error)
        test_errors.append(test_error)
    plot(iterations, train_errors, test_errors)
