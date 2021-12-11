from matplotlib import pyplot as plt
import numpy as np

# create network builder
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_dx = None
        self.err_curve = []


    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_dx):
        self.loss = loss
        self.loss_dx = loss_dx

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, learning_d):
        # sample dimension first
        size = len(x_train)

        # training loop
        for i in range(1, epochs+1):
            inds = np.random.choice(range(size), size, replace=False)
            err = 0
            lrt = learning_rate / (1 + (learning_rate / learning_d) * i)

            for j in inds:
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_dx(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, lrt)

            # calculate average error on all samples
            err /= size
            self.err_curve.append((i, err))
            if i % 5 == 0:
                print('Epoch %d/%d   MSE error=%f' % (i, epochs, err))

    # plot learning curve
    def plot(self):
        plt.plot([i[0] for i in self.err_curve], [i[1] for i in self.err_curve])
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Learning Curve')
        return plt
