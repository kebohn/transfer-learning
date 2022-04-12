from abc import (
    ABC,
    abstractmethod
)


class BaseModel(ABC):
    def __init__(self, device):
        self.device = device

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass
