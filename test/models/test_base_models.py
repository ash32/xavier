import os

import numpy as np
import pytest

from xavier.helpers import normalize
from xavier.models.buy_and_hold import BuyAndHoldModel
from xavier.models.const_distribution import ConstantDistributionModel


@pytest.fixture
def closing_price_rel_vec(request):
    return np.load(os.path.join(request.fspath.dirname, 'data', 'closing_prices_rel_vec.npy'))


def test_const_dist_model(closing_price_rel_vec):
    model = ConstantDistributionModel()

    periods, num_assets = closing_price_rel_vec.shape
    expected_weights = np.ones(num_assets) / num_assets
    for i in range(periods):
        weights = model.get_distribution_weights(closing_price_rel_vec[i], expected_weights)
        np.testing.assert_equal(weights, expected_weights)


def test_buy_and_hold(closing_price_rel_vec):
    model = BuyAndHoldModel()

    periods, num_assets = closing_price_rel_vec.shape
    prev_weights = np.ones(num_assets) / num_assets
    expected_weights = normalize(np.cumprod(closing_price_rel_vec, axis=0), axis=1)
    for i in range(periods):
        weights = model.get_distribution_weights(closing_price_rel_vec[i], prev_weights)
        np.testing.assert_allclose(weights, expected_weights[i])

        prev_weights = weights
