from abc import ABC, abstractmethod


class BaseExplorationStrategy(ABC):
    @abstractmethod
    def get_next_eps(self, eps: float) -> float:
        pass
    
    def __call__(self, eps: float) -> float:
        return self.get_next_eps(eps)
    

class LinearDecayStrategy(BaseExplorationStrategy):
    def __init__(self, min_eps=0.01, max_eps=1, exploration_progress=1e6) -> None:
        super().__init__()
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.exploration_progress = exploration_progress
    
    def get_next_eps(self, eps: float) -> float:
        return max(eps - (self.max_eps - self.min_eps) / self.exploration_progress, self.min_eps)
    

class ExponentialDecayStrategy(BaseExplorationStrategy):
    def __init__(self, min_eps=0.01, decay_rate=0.99) -> None:
        super().__init__()
        self.min_eps = min_eps
        self.decay_rate = decay_rate
        
    def get_next_eps(self, eps: float) -> float:
        return max(eps * self.decay_rate, self.min_eps)
