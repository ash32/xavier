import numpy as np
import pytest

from xavier.models.buy_and_hold import BuyAndHoldModel
from xavier.models.const_distribution import ConstantDistributionModel


@pytest.fixture
def closing_price_rel_vec():
    import os
    print(os.getcwd())
    assert os.getcwd() == 'abc'
    return np.load('../test_data/closing_prices_rel_vec.npy')


def test_const_dist_model(closing_price_rel_vec):
    model = ConstantDistributionModel()

    periods, num_assets = closing_price_rel_vec.shape
    expected_weights = np.ones(num_assets+1) / (num_assets+1)
    for i in range(periods):
        weights = model.get_distribution_weights(closing_price_rel_vec[i], None)
        assert weights == expected_weights


def test_buy_and_hold(closing_price_rel_vec):
    model = BuyAndHoldModel()

    num_assets = closing_price_rel_vec.shape[0]
    initial_weights = np.ones(num_assets+1) / (num_assets+1)