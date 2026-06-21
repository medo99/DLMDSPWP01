import pandas as pd


class DataLoader:
    def __init__(self):
        self.train_df = None
        self.ideal_df = None
        self.test_df = None

    def load_data(self):
        self.train_df = pd.read_csv("data/raw/train.csv")
        self.ideal_df = pd.read_csv("data/raw/ideal.csv")
        self.test_df = pd.read_csv("data/raw/test.csv")

        return self.train_df, self.ideal_df, self.test_df