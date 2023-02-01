import pandas as pd

from sklearn.model_selection import train_test_split


def read_dataset(path):
    return pd.read_csv(path, encoding='utf-8')

def train_test_split_dataset(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return train_df, test_df


