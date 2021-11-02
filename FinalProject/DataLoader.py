import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class DataLoader:

    ## initiate and import data
    def __init__(self, path, sheet='Message 1'):
        self.data_path = path
        self.data = pd.read_excel(self.data_path, sheet)
        self.data = self.data.sort_values(['encounterId', 'Unnamed: 0'])

    ## Randomly select a small portion of encounters for testing
    def random_select(self, n=500):
        np.random.seed(123)
        encounterId = self.data['encounterId'].unique()
        max = self.data['encounterId'].nunique()
        if n > max:
            raise ValueError('Number of selected datapoints exceeds the maximum number of the entire dataset.')
        else:
            rand_inds = np.random.choice(encounterId, n, replace=False)
            self.data = self.data[self.data['encounterId'].isin(rand_inds)]

    ## Plot data to inspect distribution
    def plot(self):
        sizes = self.data.groupby('encounterId').size()
        plt.hist(sizes, bins=range(100))
        plt.xlim(0, 100)
        plt.xticks(range(0, 101, 10))
        plt.xlabel('Number of Messages')
        plt.title('Distribution of Lengths of Conversations, <= 100-message shown')
        return plt

    ## Create and assign outcome (dropout) values
    def to_dropout(self, cutoff=6):
        sizes = self.data.groupby('encounterId').size()
        sizes.name = 'encounterLen'
        self.data = self.data.join(sizes, on='encounterId')
        self.data['dropout'] = self.data['encounterLen'].apply(lambda x: 1 if x <= cutoff else 0)