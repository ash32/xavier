from xavier.models.base import BaseModel

import numpy as np


class ConstantDistributionModel(BaseModel):
    def get_distribution_weights(self, closing_price_vec, prev_weight_vec):
        return np.ones(prev_weight_vec.size) / prev_weight_vec.size
