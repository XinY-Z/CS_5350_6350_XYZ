import pandas as pd
import numpy as np
from Plotter import plot_regressor
import sys


## load a csv file
def load_csv(filepath):
    features = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']
    outcome = ['Slump']
    dataset = pd.read_csv(filepath, names=features+outcome)
    x = dataset[features]
    y = dataset[outcome]
    return x, y

def gradient_descent(x, y, alpha, tolerance, max_iter):
    m = y.shape[0]
    w = [0] * x.shape[1]
    past_j = []
    past_w = [w]
    converge = 999
    iteration = 1
    while converge >= tolerance:
        if iteration >= max_iter:
            print('Warning: Model connot converge. Reached max iteration.')
            break
        error, j = cost(x, y, w)
        past_j.append(j)
        gradient = alpha * (1/m) * np.dot(x.T, error)
        w = w - gradient
        past_w.append(w)
        converge = np.sqrt(np.dot(gradient.T, gradient))
        iteration += 1
    return past_w, past_j

def stochastic_gradient_descent(x, y, alpha, tolerance, max_iter):
    m = y.shape[0]
    w = [0] * x.shape[1]
    past_j = []
    past_w = [w]
    converge = 999
    iteration = 1
    while converge >= tolerance:
        if iteration >= max_iter:
            print('Warning: Model connot converge. Reached max iteration.')
            break
        rand_ind = np.random.choice(range(m), m, replace=False)
        for ind in rand_ind:
            error, j = cost(x, y, w)
            past_j.append(j)
            gradient = alpha * (1/m) * np.dot(x.iloc[ind].T, error[ind])
            w = w - gradient
            past_w.append(w)
            converge = np.sqrt(np.dot(gradient.T, gradient))
            if converge >= tolerance:
                break
        iteration += 1
    return past_w, past_j

def cost(x, y, w):
    m = y.shape[0]
    prediction = np.dot(x, w)
    error = prediction - y['Slump']
    j = 1 / (2 * m) * np.dot(error.T, error)
    return error, j

def driver(x, y, r, tol, max_iteration, algorithm):
    w_list, j_list = algorithm(x, y, r, tol, max_iteration)

    ## adjust learning rate
    while any(j >= 100 for j in j_list):
        print('Adjusting learning rate...')
        r = r / 2
        w_list, j_list = gradient_descent(train_x, train_y, r, tol, max_iteration)
    while len(w_list) >= max_iteration:
        print('Adjusting learning rate...')
        r = r * 1.5
        w_list, j_list = gradient_descent(train_x, train_y, r, tol, max_iteration)
    return w_list, j_list, r


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    train_x, train_y = load_csv(train_file)
    test_x, test_y = load_csv(test_file)
    try:
        r = float(sys.argv[3])
        tol = float(sys.argv[4])
        max_iteration = int(sys.argv[5])
    except IndexError:
        r = 1.0
        tol = 1e-6
        max_iteration = 20000

    print('Using gradient descent...')
    w_list, j_list, r = driver(train_x, train_y, r, tol, max_iteration, gradient_descent)
    final_w = w_list[-1]
    test_error, test_j = cost(test_x, test_y, final_w)
    print(f'Converged at iteration {len(j_list)}')
    print(f'Cost of the test set is {round(test_j, 3)}')
    print(f'Coefficients are {final_w}')
    print(f'Learning rate is {r}')
    plot_sel = input('Do you want to plot cost trajectory? (Y/N)')
    if plot_sel == 'Y':
        plot_regressor(range(len(j_list)), j_list)

    print('Using stochastic gradient descent...')
    w_list2, j_list2, r2 = driver(train_x, train_y, r, tol, max_iteration, stochastic_gradient_descent)
    final_w2 = w_list2[-1]
    test_error2, test_j2 = cost(test_x, test_y, final_w)
    print(f'Converged at iteration {len(j_list2)}')
    print(f'Cost of the test set is {round(test_j2, 3)}')
    print(f'Coefficients are {final_w2}')
    print(f'Learning rate is {r2}')
    plot_sel2 = input('Do you want to plot cost trajectory? (Y/N)')
    if plot_sel2 == 'Y':
        plot_regressor(range(len(j_list2)), j_list2)
