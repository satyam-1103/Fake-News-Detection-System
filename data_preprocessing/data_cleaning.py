import warnings
warnings.filterwarnings("ignore","\nPyarrow", DeprecationWarning)

import pandas as pd

dataset_1 = pd.read_csv('../dataset/news_articles.csv')

def removeDuplicates():
    dataset_1.drop_duplicates(inplace=True)

def removeMissing():
    dataset_1.dropna(axis=0, inplace=True)

removedMissing = removeMissing()
removedDuplicates = removeDuplicates()

dataset_1.to_csv("../dataset/cleaned_dataset1.csv", index=False)


