from abc import ABC, abstractmethod
from typing import OrderedDict


class BaseClient(ABC):
    @abstractmethod
    def update_weights(self) -> None:
        pass
    
    @abstractmethod
    def receive_weight(self, state_dict: OrderedDict) -> None:
        pass
