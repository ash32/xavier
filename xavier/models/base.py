from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def get_distribution_weights(self, closing_price_vec, prev_weight_vec):
        pass


class TrainableModel(BaseModel):
    pass
