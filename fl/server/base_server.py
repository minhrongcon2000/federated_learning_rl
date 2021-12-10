from abc import ABC, abstractmethod


class BaseServer(ABC):
    @abstractmethod
    def aggregate(self) -> None:
        pass
    
    @abstractmethod
    def broadcast(self) -> None:
        pass