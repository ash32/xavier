from .base import FeatureGenerator
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class RelativePricingFeatureGenerator(FeatureGenerator):
    def __init__(self, df_path, train_start, train_end, test_start, test_end, num_periods=50):
        self.num_periods = num_periods

        df = pd.read_csv(df_path, index_col=0)
        df.index = df.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        df.index.names = ['date']

        test_start = test_start - num_periods * timedelta(minutes=30)

        self.train_df = df.loc[train_start:train_end]
        self.test_df = df.loc[test_start:test_end]

    @staticmethod
    def geometric_sample_weights(n, q):
        u = np.empty((n,))
        u[:] = 1-q
        return np.cumprod(u)[::-1] * q

    def get_

    def generate_training_batch(self, sample_bias=0.00005, batch_size=50):
        total = self.train_df.shape[0]
        geo_size = total - batch_size - self.num_periods + 1

        weights = self.geometric_sample_weights(geo_size, sample_bias)
        weights = weights/np.linalg.norm(weights, ord=1)  # normalize weights

        end_idx = np.random.choice(weights.size, p=weights)+1
        start_idx = end_idx - batch_size - self.num_periods

        vals = self.train_df.iloc[start_idx:end_idx]

        # Convert to numpy array
        shape = vals.shape
        vals = vals.as_matrix().reshape((shape[0], shape[1]/3, 3))
        vals = vals / vals[-1, :, 0].reshape((1, -1, 1))  # normalize by the most recent closing price for all assets

        return vals


