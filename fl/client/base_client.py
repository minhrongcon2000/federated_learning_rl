from abc import ABC, abstractmethod


class BaseClient(ABC):
    @abstractmethod
    def update_weights(self) -> None:
        pass
