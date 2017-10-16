from .base import FeatureGenerator
import pandas as pd
import datetime
import numpy as np


class RelativePricingFeatureGenerator(FeatureGenerator):
    def __init__(self, df_path, train_start, train_end, test_start, test_end):
        df = pd.read_csv(df_path, index_col=0)
        df.index = df.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

        self.train_df = df.loc[train_start:train_end]
        self.test_df = df.loc[test_start:test_end]

    def geometric_sample_weights(self, n, ):

    def generate_training_batch(self, sample_bias=0.00005):
        weights =
