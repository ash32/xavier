import unittest

from datetime import datetime

from src.features.poloniex import RelativePricingFeatureGenerator


class FeatureTest(unittest.TestCase):

    def setUp(self):
        fmt = "%Y-%m-%d %H:%M:%S"
        train_start = datetime.strptime('2017-10-18 21:00:00', fmt)
        train_end = datetime.strptime('2017-10-19 02:00:00', fmt)
        test_start = datetime.strptime('2017-10-19 02:00:00', fmt)
        test_end = datetime.strptime('2017-10-19 05:00:00', fmt)
        self.gen = RelativePricingFeatureGenerator('test_df.csv',  num_periods=3)

    def test_