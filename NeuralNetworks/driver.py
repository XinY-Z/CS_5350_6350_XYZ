import pandas as pd
import numpy as np
import sys

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import sigmoid, sigmoid_dx
from losses import mse, mse_dx

# load data
def load_csv(filepath):
    features = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    outcome = ['Outcome']
    dataset = pd.read_csv(filepath, names=features + outcome)
    x = dataset[features]
    y = dataset[outcome]
    x = x.to_numpy().reshape(x.shape[0], 1, x.shape[1])
    y = y.to_numpy().reshape(y.shape[0], 1, y.shape[1])
    return x, y


# run the model
def run(train_file, test_file, nodes, epoch, init_option, learning_rate, learning_d):

    # load data
    train_x, train_y = load_csv(train_file)
    test_x, test_y = load_csv(test_file)

    nfeature = train_x.shape[2]
    nout = train_y.shape[2]

    # set random seed
    np.random.seed(123)

    # network
    net = Network()
    net.add(FCLayer(nfeature, nodes, init_option))         # hidden layer 1
    net.add(ActivationLayer(sigmoid, sigmoid_dx))
    net.add(FCLayer(nodes, nodes, init_option))            # hidden layer 2
    net.add(ActivationLayer(sigmoid, sigmoid_dx))
    net.add(FCLayer(nodes, nout, init_option))             # output layer

    # train
    net.use(mse, mse_dx)
    net.fit(train_x, train_y, epochs=epoch, learning_rate=learning_rate, learning_d=learning_d)

    # evaluate
    train_out = net.predict(train_x)
    test_out = net.predict(test_x)

    # calculate accuracy
    train_loss = mse(train_y, train_out)
    test_loss = mse(test_y, test_out)

    net.plot()
    print(f'Training error: {train_loss}')
    print(f'Test error: {test_loss}')


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    nodes = int(sys.argv[3])
    epoch = int(sys.argv[4])
    init_option = sys.argv[5]
    try:
        learning_rate = float(sys.argv[6])
        learning_d = float(sys.argv[7])
    except IndexError:
        learning_rate = 0.1
        learning_d = 0.1

    run(train_file, test_file, nodes, epoch, init_option, learning_rate, learning_d)

train_file = './NeuralNetworks/bank-note/train.csv'
test_file = './NeuralNetworks/bank-note/test.csv'