import pandas as pd
from matplotlib import pyplot as plt

data_path = r'C:\Users\XinZ\Box\SafeUT_Data\Final_Anonymization\FINAL_ANONYMIZED_SAFEUT.xlsx'
data = pd.read_excel(data_path, 'Message 1')
data = data.sort_values(['encounterId', 'Unnamed: 0'])

## The distribution follows inverse power law distribution.
sizes = data.groupby('encounterId').size()
def plot(sizes):
    plt.hist(sizes, bins=range(100))
    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))
    plt.xlabel('Number of Messages')
    plt.title('Distribution of Lengths of Conversations, <= 100-message shown')
    return plt

## assign message counts to each encounter
sizes.name = 'encounterLen'
data = data.join(sizes, on='encounterId')

## define criterion for dropout and add new column to the dataset (criterion = # message <= 6)
data['dropout'] = data['encounterLen'].apply(lambda x: 1 if x <= 6 else 0)
