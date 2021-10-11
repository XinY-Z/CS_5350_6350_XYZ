import matplotlib.pyplot as plt

def plot(niters, train_errors, test_errors):
    plt.plot(niters, train_errors, label='Training Error')
    plt.plot(niters, test_errors, label='Test Error')
    plt.legend()
    plt.show()
