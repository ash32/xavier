from xavier.models.base import BaseModel
from xavier.helpers import normalize


class BuyAndHoldModel(BaseModel):
    def get_distribution_weights(self, closing_price_vec, prev_weight_vec):
        return normalize(closing_price_vec * prev_weight_vec)
